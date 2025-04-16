import torch
import contextlib
import os
from torch.utils._python_dispatch import TorchDispatchMode
import time


class DispatchLog(TorchDispatchMode):
    def __init__(self):
        super(DispatchLog, self).__init__()
        self.flops = 0
        self.memory = 0
 
 
    def get_shape(self,x):
        if isinstance(x, torch.Tensor):
            # need count broadcast
            memory_input = x.numel() * x.element_size()
            self.memory += memory_input
            return x.shape
        if isinstance(x, list) or isinstance(x, tuple):
            shape_list = []
            for t in x:
                if isinstance(t,  torch.Tensor):
                    memory_input = t.numel()*t.element_size()
                    self.memory += memory_input
                    shape_list.append(t.shape)
                else:
                    shape_list.append(type(t))
            return shape_list
 
        return type(x)
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        op = func.name()
        output = func(*args, **(kwargs or {}))
        if "view" in op or "transpose" in op or "slice" in op or "split" in op or "unsqueeze" in op or "permute" in op or "expand" in op or "aten::select" in op or op == "aten::t" or "detach" in op or op == "aten::_local_scalar_dense" or op=="aten::lift_fresh":
            return output
        args_shapes = self.get_shape(args) if isinstance(args, torch.Tensor) else  (tuple(self.get_shape(arg) for arg in args))
        kwargs_shapes = {k: self.get_shape(v) for k, v in (kwargs or {}).items()}
        if isinstance(output, torch.Tensor):
            output_shapes = self.get_shape(output)
        elif isinstance(output, list) or isinstance(output, tuple):
            output_shapes = tuple(self.get_shape(o) for o in output)
        if op == "aten::bmm":
            self.flops += 2 * args[0].shape[0]* args[0].shape[1] * args[0].shape[2] * args[1].shape[2]
        if op == "aten::addmm":
            self.flops += (2 * args[1].shape[0]*args[1].shape[1] * args[2].shape[-1]+ args[1].shape[0]*args[2].shape[1])
        if op == "aten::mm":
            self.flops += (2 * args[0].shape[0]*args[0].shape[1] * args[1].shape[-1])
        if op == "aten::convolution":
            assert args[0].dim()==4 and args[1].dim()==4
            N = args[0].shape[0]
            C_i = args[0].shape[1]
            H_i = args[0].shape[2]
            W_i = args[0].shape[3]
            K_h = args[1].shape[2]
            K_w = args[1].shape[3]
            K_1 = args[1].shape[1]
            H_o = output.shape[2]
            W_o = output.shape[3]
            C_o = output.shape[1]
            self.flops += 2* H_o * W_o * N * K_h * K_w * K_1 * C_o
            if isinstance(args[2],torch.Tensor):
                # bias is another kernel
                self.memory += N * C_o * H_o * W_o * 2
                self.flops += N * C_o * H_o * W_o
        if op == "aten::max_pool2d_with_indices":
            N = output[0].shape[0]
            C = output[0].shape[1]
            H = output[0].shape[2]
            W = output[0].shape[3]
            K_h = args[1][0]
            K_w = args[1][-1]
            self.flops += N * C * H * W * (K_h * K_w - 1)
        if "batch_norm" in op:
            self.flops += 4*args[0].numel()
        if op == "aten::native_layer_norm":
            if args[0].dtype == torch.float16:
                self.flops += 5*args[0].numel()
            else:
                self.flops += 5*args[0].numel() * 15 # suppose fp32 TFLOPS is 15x slower than FP16 according to spec
        if op == "aten::_softmax":
            self.flops += 4*args[0].numel()
        print(f"{op}|{self.flops}|{self.memory}|{args_shapes}", flush=True)
        return output

@contextlib.contextmanager
def context_func(profiling_enabled, device, fuser_mode='none', schedule_disable='no', total_iter=None):
    calculate_flops = os.environ.get("Calculate_Flops", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
    
    profile_activity = [torch.profiler.ProfilerActivity.CPU]
    if device == "xpu":
        profile_activity.append(torch.profiler.ProfilerActivity.XPU)
    elif device == "cuda":
        profile_activity.append(torch.profiler.ProfilerActivity.CUDA)

    if schedule_disable == "yes":
        schedule = None
    elif total_iter != None:
        middle_iter = total_iter // 2 + 4 if total_iter >= 10 else total_iter
        schedule = torch.profiler.schedule(wait=middle_iter-3, warmup=3, active=1)
    else:
        schedule = torch.profiler.schedule(wait=6, warmup=3, active=1)

    if profiling_enabled:
        with torch.profiler.profile(activities=profile_activity, schedule=schedule) as prof:
            yield prof
    elif calculate_flops:
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        with DispatchLog() as calculator:
            yield calculator
        print("memory:", calculator.memory, flush=True)
        print("flops:",calculator.flops, flush=True)
    else:
        with contextlib.nullcontext(None):
            yield

    if profiling_enabled:
        save_profile(prof, device)
        print("---- save profile success")

def save_profile(prof, device):
    import pathlib
    import os
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    torch.save(prof.key_averages().table(sort_by="self_{}_time_total".format(device), row_limit=100000),
        timeline_dir+'profile.pt')
    torch.save(prof.key_averages(group_by_input_shape=True).table(),
        timeline_dir+'profile_detail.pt')
    #torch.save(prof.key_averages().table(sort_by="id", row_limit=100000),
    #    timeline_dir+'profile_detail_withId.pt')
    prof.export_chrome_trace(timeline_dir+"trace.json")

