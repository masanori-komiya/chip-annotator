import json
import sys


def main():
    import torch
    import mobile_sam
    result = dict(python=sys.version, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), architectures=torch.cuda.get_arch_list())
    if torch.cuda.is_available():
        result['gpu'] = torch.cuda.get_device_name(0)
        result['capability'] = torch.cuda.get_device_capability(0)
        x = torch.ones((16, 16), device='cuda')
        result['cuda_matmul'] = float((x @ x).sum())
        torch.cuda.synchronize()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
