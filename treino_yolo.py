from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    model.train(
        data="Column.v2i.yolov8/data.yaml",
        epochs=250,
        device=0,
        imgsz=512,
        batch=24,
        patience=50,
        cache=True,
        workers=2
    )

if __name__ == "__main__":
    main()