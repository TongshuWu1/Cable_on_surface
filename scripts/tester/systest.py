import pyzed.sl as sl

def main():
    zed = sl.Camera()
    init = sl.InitParameters()

    status = zed.open(init)
    print("Open status:", status)

    if status == sl.ERROR_CODE.SUCCESS:
        info = zed.get_camera_information()
        print("Camera opened")
        print("Serial number:", info.serial_number)
        zed.close()
    else:
        print("Failed to open camera")

if __name__ == "__main__":
    main()