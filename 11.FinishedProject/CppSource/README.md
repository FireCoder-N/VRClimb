The CppSource folder contains the C++ headers (`/Public/*.h`) as well as the implementations (`/Private/*.cpp`) of various features of the Unreal Engine Project.

Note that among those files, some are legacy implementations, not needed in the final project. Below, there is a list of all the files are actually used in the project, as well as a short descritption for each one of them:

- `OpenVRManager.cpp`: Manage the tracking of Vive VR trackers and return their position in real-life worldspace coordinates.

- `MoonController.cpp`: Manage the appearence of the moon of the virtual environment during the night.

- `Importer.cpp`: Create and Manage an import task in order to import the reconstructed wall mesh as a UE5 static mesh asset.

- `IRLPointsSave.cpp`: **May** be used to track the four corners of the climbing wall. As of the stable version of the project, those coordinates where recorded once and now they are hardcoded manually, so that the file is technically not required.

*Note: In order to run this project, it is required to download the `openvr_api.dll` and place it inside the Binaries folder.*