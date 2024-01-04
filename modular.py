import subprocess

def train_yolov7(workers, device, batch_size, data, img_size, cfg, weights, name, hyp):
    cmd = [
        'python', 'train.py',
        '--workers', str(workers),
        '--device', str(device),
        '--batch-size', str(batch_size),
        '--data', data,
        '--img', str(img_size), str(img_size),
        '--cfg', cfg,
        '--weights', weights,
        '--name', name,
        '--hyp', hyp
    ]

    subprocess.run(cmd)

# Example usage
train_yolov7(workers=8, device='cpu', batch_size=2, data='data/custom.yaml',
              img_size=640, cfg='cfg/training/yolov7.yaml', weights='./yolov7.pt', name='yolov7', 
              hyp='data/hyp.scratch.p5.yaml')


