# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
import os
import platform
import subprocess
import time

import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from src.logger import CSVLogger
from src.models import get_model_by_string
from src.models.model_utils import cnt_parameters
from src.data.preprocessing import get_preprocessing

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
TRT_WORKSPACE = 2 << 30    # 2 GB


def _parse_args():
    parser = argparse.ArgumentParser(description='Time inference')
    parser.add_argument('--model',
                        type=str,
                        default='beyer_mod_relu_depth_126_48',
                        help=('network to use, '
                              'default: beyer_mod_relu_depth_126_48'))
    parser.add_argument('--device',
                        type=str,
                        default='gpu',
                        choices=['gpu', 'cpu'],
                        help='device to use, default: gpu')
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        help='batch size for inference, default: 20')
    parser.add_argument('--floatx',
                        type=int,
                        nargs='+',
                        choices=[32, 16],
                        default=[32, 16],
                        help=('tensorrt floatx for inference, '
                              'default: [32, 16]'))
    parser.add_argument('--n_runs',
                        type=int,
                        default=100,
                        help='number of runs to average over, default: 100')
    parser.add_argument('--n_runs_warmup',
                        type=int,
                        default=5,
                        help=('number of additional initial runs without '
                              'timing, default: 5'))
    args = parser.parse_args()
    return args


def _get_engine(onnx_filepath,
                engine_filepath,
                force_rebuild=True,
                floatx=32,
                batch_size=1):
    if not os.path.exists(engine_filepath) or force_rebuild:
        print("Building engine using onnx2trt")
        if floatx == 32:
            print("... this may take a while")
        else:
            print("... this may take -> AGES <-")
        cmd = f'{os.path.expanduser("onnx2trt")} {onnx_filepath}'
        cmd += f' -d {floatx}'    # 16: float16, 32: float32
        cmd += f' -b {batch_size}'    # batchsize
        cmd += ' -v'    # verbose
        cmd += ' -l'    # list layers
        cmd += f' -w {TRT_WORKSPACE}'   # workspace size mb
        cmd += f' -o {engine_filepath}'

        try:
            print(cmd)
            out = subprocess.check_output(cmd,
                                          shell=True,
                                          stderr=subprocess.STDOUT,
                                          universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print("onnx2trt failed:", e.returncode, e.output)
            raise
        print(out)

    print(f"Loading engine: {engine_filepath}")
    with open(engine_filepath, "rb") as f, trt.Runtime(
            TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def _alloc_buf(engine):
    # input binding
    shape = trt.volume(engine.get_binding_shape(0))  # * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(0))
    in_cpu = cuda.pagelocked_empty(shape, dtype)
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)

    # output binding
    shape = trt.volume(engine.get_binding_shape(1))  # * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(1))
    out_cpu = cuda.pagelocked_empty(shape, dtype)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)

    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream


def _time_inference_pytorch(model,
                            inputs,
                            device,
                            n_runs=100,
                            n_runs_warmup=5):
    timings = []
    outputs = []
    with torch.no_grad():
        for i in range(n_runs + n_runs_warmup):
            start_time = time.time()

            # copy to gpu
            input_gpu = torch.from_numpy(inputs[i]).to(device)

            # model forward pass
            out_pytorch = model(input_gpu)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            out_pytorch = out_pytorch.cpu()

            if i >= n_runs_warmup:
                timings.append(time.time() - start_time)
                outputs.append(out_pytorch.cpu().numpy())

    return np.array(timings), np.array(outputs)


def _time_inference_tensorrt(onnx_filepath,
                             inputs,
                             batch_size=1,
                             floatx=32,
                             n_runs=30,
                             n_runs_warmup=5):
    # create engine
    trt_filepath = os.path.splitext(onnx_filepath)[0] + '.trt'

    engine = _get_engine(onnx_filepath, trt_filepath,
                         batch_size=batch_size, floatx=floatx,
                         force_rebuild=True)
    context = engine.create_execution_context()

    # allocate memory on gpu
    in_cpu, out_cpu, in_gpu, out_gpu, stream = _alloc_buf(engine)

    timings = []
    outputs = []
    for i in range(n_runs + n_runs_warmup):
        start_time = time.time()

        # copy to gpu
        cuda.memcpy_htod(in_gpu, inputs[i])

        # model forward pass
        context.execute(batch_size=batch_size,
                        bindings=[int(in_gpu), int(out_gpu)])
        # stream.synchronize()

        # copy back to cpu
        cuda.memcpy_dtoh(out_cpu, out_gpu)

        if i >= n_runs_warmup:
            timings.append(time.time() - start_time)
            outputs.append(out_cpu.copy())

    return np.array(timings), np.array(outputs)


if __name__ == '__main__':
    # parameters --------------------------------------------------------------
    plot_timings = False

    args = _parse_args()

    logger = CSVLogger('./time_inference_runtimes.csv')
    pytorch_version = torch.__version__
    tensorrt_version = trt.__version__

    if args.device == 'gpu':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        device = torch.device('cuda')
        time_tensortt = True
    else:
        device = torch.device('cpu')
        time_tensortt = False
    time_pytorch = True

    if plot_timings:
        import matplotlib.pyplot as plt
        plt.figure()

    # create random input -----------------------------------------------------
    preprocess, _, _ = get_preprocessing(args.model)

    img_shape = (100, 100)
    inputs = []
    n = args.n_runs + args.n_runs_warmup
    for _ in range(n*args.batch_size):
        img = np.random.randint(0, 40000, size=img_shape, dtype='uint16')
        img = preprocess(img)

        inputs.append(img)
    inputs = np.array(inputs)
    inputs.shape = (n, args.batch_size) + inputs.shape[-3:]

    # get model ---------------------------------------------------------------
    model = get_model_by_string(args.model, device=device)
    model.eval()

    n_parameters = cnt_parameters(model)['total']

    # time inference using PyTorch --------------------------------------------
    if time_pytorch:
        timings_pytorch, outputs_pytorch = _time_inference_pytorch(
            model,
            inputs,
            device,
            n_runs=args.n_runs,
            n_runs_warmup=args.n_runs_warmup
        )

        print(f'runs per second pytorch: {np.mean(1/timings_pytorch):0.4f} ± '
              f'{np.std(1/timings_pytorch):0.4f}')

        logger.write_logs({'model': args.model,
                           'n_parameters': n_parameters,
                           'hostname': platform.node(),
                           'runtime_engine': f'pytorch_{pytorch_version}',
                           'floatx': 32,
                           'batch_size': args.batch_size,
                           'device': args.device,
                           'runs_per_second_mean': np.mean(1/timings_pytorch),
                           'runs_per_second_std': np.std(1/timings_pytorch),
                           'n_runs': args.n_runs,
                           'abs_err_to_pytorch_mean': -1,
                           'abs_err_to_pytorch_std': -1,
                           })

        if plot_timings:
            plt.plot(1 / timings_pytorch, label='pytorch')

    # time inference using TensorRT -------------------------------------------
    if time_tensortt:
        dummy_input = inputs[0]
        input_names = ['input']
        output_names = ['output']
        opset_version = 10
        onnx_filepath = './model.onnx'

        torch.onnx.export(model,
                          torch.tensor(dummy_input, device=device),
                          onnx_filepath,
                          export_params=True,
                          input_names=input_names,
                          output_names=output_names,
                          do_constant_folding=True,
                          verbose=True,
                          opset_version=opset_version)

        for fx in args.floatx:
            timings_tensorrt, outputs_tensorrt = _time_inference_tensorrt(
                onnx_filepath,
                inputs,
                batch_size=args.batch_size,
                floatx=fx,
                n_runs=args.n_runs,
                n_runs_warmup=args.n_runs_warmup
            )

            print(f'runs per second tensorrt float{fx}: '
                  f'{np.mean(1 / timings_tensorrt):0.4f} ± '
                  f'{np.std(1 / timings_tensorrt):0.4f}')

            # check for similar results
            if time_pytorch:
                outputs_tensorrt.shape = outputs_pytorch.shape

                outputs_pytorch.shape = (-1, outputs_pytorch.shape[-1])
                outputs_tensorrt.shape = outputs_pytorch.shape

                err = np.abs(outputs_pytorch - outputs_tensorrt).sum(axis=-1)
                err_mean = err.mean()
                err_std = err.std()
                print(f"output difference: {err_mean} ± {err_std}")
            else:
                err_mean = -1
                err_std = -1

            logger.write_logs({
                'model': args.model,
                'n_parameters': n_parameters,
                'hostname': platform.node(),
                'runtime_engine': f'tensorrt_{tensorrt_version}',
                'floatx': fx,
                'batch_size': args.batch_size,
                'device': args.device,
                'runs_per_second_mean': np.mean(1/timings_tensorrt),
                'runs_per_second_std': np.std(1/timings_tensorrt),
                'n_runs': args.n_runs,
                'abs_err_to_pytorch_mean': err_mean,
                'abs_err_to_pytorch_std': err_std,
            })

            if plot_timings:
                plt.plot(1/timings_tensorrt, label=f'tensorrt_{fx}')

    # plot results ------------------------------------------------------------
    if plot_timings:
        plt.xlabel("run")
        plt.ylabel("runs per second [Hz]")
        plt.legend()
        plt.title(f"Inference time: {args.model}")
        plt.show()
