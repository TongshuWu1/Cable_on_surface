import cv2
import pyzed.sl as sl

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera:", status)
        return

    image = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    print("Press q to quit")

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            cv2.imshow("ZED Left Image", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()