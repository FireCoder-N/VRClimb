These python scripts are called by unreal engine in order to reconstruct the mesh of the climbing wall.

Following the notation of the meain documentation, as well as the Editor Utility Widget/ Menu (EUW_UI) within Unreal Engine:

- The user first selects if they prefer the default homography stitcher from OpenCV (stitch_alt.py) or the custome one (stitch.py). The stitching process is executed when the first button `Stitch Panorama` is clicked. The file `post_process.py` is also automatically called after the stitching in order to clean-up and 'flatten' the resulting panorama.

- The second button (`Crop Panorama`) calls the `crop.py` file to crop the panorma into a clean image. For the cropping a simple `Tkinter` app is used.

- The third button (`Generate Mesh`) executes the `auto_mesh.py` file, which -after running for a couple of minutes- utilizes yolo, MiDaS, opencv and open3d to reconstruct a mesh based on the panorama image from before

*Note: In order to run this project, it is required to download the `openvr_api.dll` and place it inside the Binaries folder.*