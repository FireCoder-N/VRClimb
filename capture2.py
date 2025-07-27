"""
capture.py

Opens a ZED window so you can see live video. User controls:

  - SPACE (key code 32):  “capture the current frame as the next image in this row (left→right)”
  -    n (key code 110):  “start a new row → capture as leftmost in the next row”
  -    q (key code 113):  “quit immediately”

Here, XX = zero-padded row index, YY = zero-padded column index in that row.
"""

import os
import cv2
import numpy as np
import pyzed.sl as sl


def make_output_dirs(base_folder):
    os.makedirs(base_folder, exist_ok=True)
    rgb_folder   = os.path.join(base_folder, "rgb")
    depth_folder = os.path.join(base_folder, "depth")
    os.makedirs(rgb_folder,   exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)
    return rgb_folder, depth_folder

def main():
    save_dir = "captures"
    rgb_folder, depth_folder = make_output_dirs(save_dir)

    # 1) Initialize ZED
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER
    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        exit(1)

    runtime = sl.RuntimeParameters()

    # Containers for RGB + Depth
    rgb_mat   = sl.Mat()
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.FILL

    # 2) Capture loop: show live video + keyboard controls
    row, col = 0, 0
    window_name = "ZED Capture (SPACE=next col, n=new row, q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("→ Controls:")
    print("   SPACE: capture → next image in same row (increments column)")
    print("     n  : capture → new row (increments row, resets column to 0)")
    print("     q  : quit immediately")
    print("")

    while True:
        # Grab a new frame from ZED
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        # Retrieve left image (BGRA) and depth (float32 in meters)
        zed.retrieve_image(rgb_mat,   sl.VIEW.LEFT)    # BGRA
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        # Convert to OpenCV formats
        rgb_ocv  = rgb_mat.get_data()                    # (H,W,4) BGRA
        rgb_bgr  = cv2.cvtColor(rgb_ocv, cv2.COLOR_BGRA2BGR)

        # Overlay text: row, col, instructions
        display = rgb_bgr.copy()
        info_text  = f"Row = {row}, Col = {col}"
        instr_text = "SPACE→save same row | n→new row | q→quit"
        cv2.putText(display, info_text,  (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(display, instr_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        # If user presses SPACE: save current frame as next in same row
        if key == 32:  # ASCII 32 = SPACE
            # Retrieve the latest depth array
            depth_ocv = depth_mat.get_data()  # shape (H,W), dtype=float32

            # Build zero-padded filenames: rXX_cYY
            prefix = f"r{row:02d}_c{col:02d}"
            rgb_path   = os.path.join(rgb_folder,   prefix + ".png")
            depth_path = os.path.join(depth_folder, prefix + ".npy")

            # Save RGB as PNG (BGR, 8-bit) & Depth as .npy (float32)
            cv2.imwrite(rgb_path, rgb_bgr)
            np.save(depth_path, depth_ocv)

            print(f"[CAPTURE] Saved → {prefix}")
            col += 1   # next column in same row

        # If user presses 'n': start a new row (increment row, reset col=0), then save that capture
        elif key == ord('n'):
            row += 1
            col = 0
            # Retrieve the latest depth array
            depth_ocv = depth_mat.get_data()

            prefix = f"r{row:02d}_c{col:02d}"
            rgb_path   = os.path.join(rgb_folder,   prefix + ".png")
            depth_path = os.path.join(depth_folder, prefix + ".npy")

            cv2.imwrite(rgb_path, rgb_bgr)
            np.save(depth_path, depth_ocv)
            print(f"[CAPTURE][NEW ROW] Saved → {prefix}")

            col += 1

        # If user presses 'q': quit
        elif key == ord('q'):
            print("Quitting capture loop.")
            break

    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
