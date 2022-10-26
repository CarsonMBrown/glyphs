from yolov5 import train


def train_yolo(yaml_path, epochs, *, img_size=640, weights='yolov5n.pt'):
    train.run(img=640,
              batch=16,
              epochs=1,
              data='dataset/yolov5_raw.yml',
              weights='yolov5s.pt',
              freeze=10,
              fliplr=0.5,
              cache='ram',
              device='cpu')
    # print(
    #     "python train.py --img 640 --batch 16 --epochs 1 --data dataset/yolov5_raw.yml --weights yolov5s.pt --freeze 10 --cache ram")

# val.run(imgsz=640, data='coco128.yaml', weights='yolov5x.pt')
# detect.run(imgsz=640)
# export.run(imgsz=640, weights='yolov5s.pt')
