// OpenVRManager.cpp
#include "OpenVRManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

bool UOpenVRManager::Initialize()
{
    vr::EVRInitError eError = vr::VRInitError_None;
    VRSystem = vr::VR_Init(&eError, vr::VRApplication_Scene);

    if (eError != vr::VRInitError_None || VRSystem == nullptr)
    {
        UE_LOG(LogTemp, Error, TEXT("OpenVR initialization failed: %s"),
            ANSI_TO_TCHAR(vr::VR_GetVRInitErrorAsEnglishDescription(eError)));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("OpenVR initialized successfully."));
    return true;
}

//void UOpenVRManager::UpdateTracking()
//{
//    if (!VRSystem) {
//        UE_LOG(LogTemp, Log, TEXT("System Abort!"));
//        return;
//    }
//
//    // Array to hold poses for all devices.
//    vr::TrackedDevicePose_t TrackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
//    VRSystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, TrackedDevicePoses, vr::k_unMaxTrackedDeviceCount);
//
//    // Loop over all possible devices.
//    for (vr::TrackedDeviceIndex_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++)
//    {
//        // Skip if device is not connected or pose is not valid.
//        if (!VRSystem->IsTrackedDeviceConnected(i) || !TrackedDevicePoses[i].bPoseIsValid)
//            continue;
//
//        // Extract position.
//        float x, y, z;
//        GetPosition(TrackedDevicePoses[i].mDeviceToAbsoluteTracking, x, y, z);
//
//        // Retrieve the device class.
//        vr::ETrackedDeviceClass deviceClass = VRSystem->GetTrackedDeviceClass(i);
//        FString DeviceClassStr;
//        switch (deviceClass)
//        {
//        case vr::TrackedDeviceClass_HMD:
//            DeviceClassStr = TEXT("HMD");
//            break;
//        case vr::TrackedDeviceClass_Controller:
//            DeviceClassStr = TEXT("Controller");
//            break;
//        case vr::TrackedDeviceClass_GenericTracker:
//            DeviceClassStr = TEXT("GenericTracker");
//            break;
//        case vr::TrackedDeviceClass_TrackingReference:
//            DeviceClassStr = TEXT("TrackingReference");
//            break;
//        case vr::TrackedDeviceClass_DisplayRedirect:
//            DeviceClassStr = TEXT("DisplayRedirect");
//            break;
//        default:
//            DeviceClassStr = TEXT("Unknown");
//            break;
//        }
//
//        // Print device index, class, and position.
//        UE_LOG(LogTemp, Log, TEXT("Device %d: %s, Position: (%.2f, %.2f, %.2f)"),
//            i, *DeviceClassStr, x, y, z);
//    }
//}

FString UOpenVRManager::DebugTracking()
{
    FString FinalLog; // String to accumulate logs

    if (!VRSystem)
        return FinalLog; // Return empty string if VRSystem is not valid

    // Array to hold poses for all devices.
    vr::TrackedDevicePose_t TrackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
    VRSystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, TrackedDevicePoses, vr::k_unMaxTrackedDeviceCount);

    // Loop over devices.
    for (vr::TrackedDeviceIndex_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++)
    {
        if (!VRSystem->IsTrackedDeviceConnected(i))
            continue;

        // Check if device is a Generic Tracker (Vive Tracker)
        if (VRSystem->GetTrackedDeviceClass(i) != vr::TrackedDeviceClass_GenericTracker) // vr::TrackedDeviceClass_HMD
            continue;

        if (!TrackedDevicePoses[i].bPoseIsValid)
            continue;

        float x, y, z;
        GetPosition(TrackedDevicePoses[i].mDeviceToAbsoluteTracking, x, y, z);

        // Append to FinalLog with improved formatting
        FString TrackerLog = FString::Printf(TEXT("Tracker %d:\n"), i);
        TrackerLog += FString::Printf(TEXT("Position: (x = %.2f, y = %.2f, z = %.2f)\n"), x, y, z);

        FQuat rotationQuat;
        GetOrientation(TrackedDevicePoses[i].mDeviceToAbsoluteTracking, rotationQuat);

        TrackerLog += FString::Printf(TEXT("Rotation (quaternion): (x = %f, y = %f, z = %f, w = %f)\n"), rotationQuat.X, rotationQuat.Y, rotationQuat.Z, rotationQuat.W);

        // Add two empty lines after each tracker log for better separation
        TrackerLog += "\n\n";

        // Append the formatted log to FinalLog string
        FinalLog += TrackerLog;

        // Also log to the Unreal Engine console
        UE_LOG(LogTemp, Log, TEXT("%s"), *TrackerLog);
    }

    return FinalLog; // Return the accumulated log
}

void UOpenVRManager::Shutdown()
{
    if (VRSystem)
    {
        vr::VR_Shutdown();
        VRSystem = nullptr;
        UE_LOG(LogTemp, Log, TEXT("OpenVR shutdown."));
    }
}

void UOpenVRManager::GetTrackerPosition(int index, float& x, float& y, float& z)
{
    if ((!VRSystem))
        return;

    // Array to hold poses for all devices.
    vr::TrackedDevicePose_t TrackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
    VRSystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, TrackedDevicePoses, vr::k_unMaxTrackedDeviceCount);

    if ((!VRSystem->IsTrackedDeviceConnected(index)) ||
        //(VRSystem->GetTrackedDeviceClass(index) != vr::TrackedDeviceClass_GenericTracker) ||
        (!TrackedDevicePoses[index].bPoseIsValid))
        return;

    GetPosition(TrackedDevicePoses[index].mDeviceToAbsoluteTracking, x, y, z);
}

void UOpenVRManager::GetTrackerOrientation(int index, FQuat& Q)
{
    if ((!VRSystem))
        return;

    // Array to hold poses for all devices.
    vr::TrackedDevicePose_t TrackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
    VRSystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, TrackedDevicePoses, vr::k_unMaxTrackedDeviceCount);

    if ((!VRSystem->IsTrackedDeviceConnected(index)) ||
        //(VRSystem->GetTrackedDeviceClass(index) != vr::TrackedDeviceClass_GenericTracker) ||
        (!TrackedDevicePoses[index].bPoseIsValid))
        return;

    //GetOrientationUnreal(TrackedDevicePoses[index].mDeviceToAbsoluteTracking, Q);

    if (index != 0)
        GetOrientationUnreal(TrackedDevicePoses[index].mDeviceToAbsoluteTracking, Q);
    else
        GetOrientationUnreal_HMD(TrackedDevicePoses[index].mDeviceToAbsoluteTracking, Q);
}


void UOpenVRManager::GetPosition(const vr::HmdMatrix34_t& mat, float& x, float& y, float& z)
{
    x = mat.m[0][3];
    y = mat.m[1][3];
    z = mat.m[2][3];
}

void UOpenVRManager::GetOrientationUnreal(const vr::HmdMatrix34_t& mat, FQuat& Q)
{
    // Step 1: Construct the rotation matrix from HmdMatrix34_t
    FMatrix RotationMatrix(
        FPlane(mat.m[0][0], mat.m[1][0], mat.m[2][0], 0),  // X Column
        FPlane(mat.m[0][1], mat.m[1][1], mat.m[2][1], 0),  // Y Column
        FPlane(mat.m[0][2], mat.m[1][2], mat.m[2][2], 0),  // Z Column
        FPlane(0, 0, 0, 1)  // Identity row for 4x4 matrix compatibility
    );

    // Step 2: Apply the coordinate system transformation (rotation adjustment).
    FMatrix AxisAdjustmentMatrix(
        FPlane(1, 0, 0, 0),
        FPlane(0, 0, 1, 0),
        FPlane(0, 1, 0, 0),
        FPlane(0, 0, 0, 1)
    );

    // Step 3: Correct 0-Pose
    const FQuat Qzero = FQuat::MakeFromEuler(FVector(-90.0f, 0.0f, 0.0f));
    FMatrix MzeroInv = FQuatRotationMatrix(Qzero.Inverse());
    // counter UE rotation
    FMatrix Rz90 = FRotationMatrix(FRotator(0, 90, 0));

    // Step 4: Apply axis corrections
    FMatrix AdjustedRotationMatrix = AxisAdjustmentMatrix * MzeroInv * RotationMatrix * AxisAdjustmentMatrix * Rz90;

    // Step 4: Convert the adjusted rotation matrix into a quaternion
    Q = FQuat(AdjustedRotationMatrix);
}

void UOpenVRManager::GetOrientationUnreal_HMD(const vr::HmdMatrix34_t& mat, FQuat& Q)
{
    // Step 1: Construct the rotation matrix from HmdMatrix34_t
    FMatrix RotationMatrix(
        FPlane(mat.m[0][0], mat.m[1][0], mat.m[2][0], 0),  // X Column
        FPlane(mat.m[0][1], mat.m[1][1], mat.m[2][1], 0),  // Y Column
        FPlane(mat.m[0][2], mat.m[1][2], mat.m[2][2], 0),  // Z Column
        FPlane(0, 0, 0, 1)  // Identity row for 4x4 matrix compatibility
    );

    // Step 2: Apply the coordinate system transformation (rotation adjustment).
    FMatrix AxisAdjustmentMatrix(
        FPlane(0, 0, -1, 0),
        FPlane(1, 0, 0, 0),
        FPlane(0, 1, 0, 0),
        FPlane(0, 0, 0, 1)
    );

    // Step 3: counter UE rotation
    FMatrix Ry = FRotationMatrix(FRotator(0, HMD_Offset, 0));

    // Step 4: Apply axis corrections
    FMatrix AdjustedRotationMatrix = AxisAdjustmentMatrix * RotationMatrix * AxisAdjustmentMatrix.GetTransposed() * Ry;

    // Step 4: Convert the adjusted rotation matrix into a quaternion
    Q = FQuat(AdjustedRotationMatrix);
}


void UOpenVRManager::GetOrientation(const vr::HmdMatrix34_t& mat, FQuat& Q)
{
    FMatrix RotationMatrix(
        FPlane(mat.m[0][0], mat.m[1][0], mat.m[2][0], 0),  // X Column
        FPlane(mat.m[0][1], mat.m[1][1], mat.m[2][1], 0),  // Y Column
        FPlane(mat.m[0][2], mat.m[1][2], mat.m[2][2], 0),  // Z Column
        FPlane(0, 0, 0, 1)  // Identity row for 4x4 matrix compatibility
    );

    Q = FQuat(RotationMatrix);
}

bool UOpenVRManager::ComputeCalibrationTransform(
    const TArray<FVector>& IRLPoints,
    const TArray<FVector>& UEPoints,
    FTransform& OutTransform)
{
    const int32 NumPoints = IRLPoints.Num();
    if (NumPoints < 3 || UEPoints.Num() != NumPoints)
    {
        UE_LOG(LogTemp, Warning, TEXT("Need at least 3 matching point pairs."));
        return false;
    }

    // --- Convert to Eigen matrices ---
    Eigen::Matrix3Xd Real(3, NumPoints);
    Eigen::Matrix3Xd Virtual(3, NumPoints);

    for (int32 i = 0; i < NumPoints; ++i)
    {
        Real(0, i) = IRLPoints[i].X;
        Real(1, i) = IRLPoints[i].Y;
        Real(2, i) = IRLPoints[i].Z;

        Virtual(0, i) = UEPoints[i].X;
        Virtual(1, i) = UEPoints[i].Y;
        Virtual(2, i) = UEPoints[i].Z;
    }

    // --- Compute centroids and zero-center ---
    Eigen::Vector3d RealCentroid = Real.rowwise().mean();
    Eigen::Vector3d VirtualCentroid = Virtual.rowwise().mean();
    Real.colwise() -= RealCentroid;
    Virtual.colwise() -= VirtualCentroid;

    // --- Covariance & SVD ---
    Eigen::Matrix3d H = Real * Virtual.transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> SVD(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = SVD.matrixU();
    Eigen::Matrix3d V = SVD.matrixV();

    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0)  // fix reflection
    {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    // --- Uniform scale (Umeyama) ---
    double varReal = 0.0;
    for (int i = 0; i < NumPoints; ++i) varReal += Real.col(i).squaredNorm();
    varReal /= static_cast<double>(NumPoints);

    double sumSingular = SVD.singularValues().sum();
    double scale = (varReal > 1e-12) ? (sumSingular / varReal) : 1.0;
    scale = 0.2 * scale;

    // --- Translation ---
    Eigen::Vector3d T = VirtualCentroid - scale * R * RealCentroid;

    // --- Convert to UE types ---
    FMatrix UE_Rotation(
        FVector(R(0, 0), R(1, 0), R(2, 0)),
        FVector(R(0, 1), R(1, 1), R(2, 1)),
        FVector(R(0, 2), R(1, 2), R(2, 2)),
        FVector::ZeroVector);

    FQuat UEQuat(UE_Rotation);
    FVector UETranslation(T(0), T(1), T(2));

    OutTransform = FTransform(UEQuat, UETranslation, FVector(scale));

    UE_LOG(LogTemp, Log, TEXT("Calibration complete: Translation=%s, Rotation=%s, Scale=%f"),
        *UETranslation.ToString(),
        *UEQuat.Rotator().ToString(),
        scale);

    return true;
}


void UOpenVRManager::CalibrateTrackers(TArray<FVector>& IrlPoints) {
    if (!VRSystem)
        return;

    // Array to hold poses for all devices.
    vr::TrackedDevicePose_t TrackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
    VRSystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, TrackedDevicePoses, vr::k_unMaxTrackedDeviceCount);

    // Loop over devices.
    for (vr::TrackedDeviceIndex_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++)
    {
        if (!VRSystem->IsTrackedDeviceConnected(i))
            continue;

        // check if device is a Generic Tracker (Vive Tracker)
        if (VRSystem->GetTrackedDeviceClass(i) != vr::TrackedDeviceClass_GenericTracker)
            continue;

        if (!TrackedDevicePoses[i].bPoseIsValid)
            continue;

        float x, y, z;
        GetPosition(TrackedDevicePoses[i].mDeviceToAbsoluteTracking, x, y, z);

        FVector TrackerIrlPosition = { x, y, z };
        IrlPoints.Add(TrackerIrlPosition);
    }
}


// Solve for affine transform A (3x3) and t (3x1) minimizing sum ||A*R_i + t - V_i||^2
bool UOpenVRManager::ComputeAffineTransform(
    const TArray<FVector>& IRLPoints,
    const TArray<FVector>& UEPoints,
    TArray<float>& OutMatrix3x4,
    bool bLog)
{
    const int N = IRLPoints.Num();
    if (N < 3 || UEPoints.Num() != N)
    {
        UE_LOG(LogTemp, Warning, TEXT("ComputeAffineTransform: need >=3 points and equal lengths."));
        return false;
    }

    // Build linear system M * x = b, where x is 12-vector [A00 A01 A02 t0 A10 ... t1 A20...t2]
    // For each point: [ x y z 1  0 0 0 0  0 0 0 0 ] * x = Vx
    //                 [ 0 0 0 0  x y z 1  0 0 0 0 ] * x = Vy
    //                 [ 0 0 0 0  0 0 0 0  x y z 1 ] * x = Vz

    Eigen::MatrixXd M(3 * N, 12);
    Eigen::VectorXd b(3 * N);

    for (int i = 0; i < N; ++i)
    {
        double x = IRLPoints[i].X;
        double y = IRLPoints[i].Y;
        double z = IRLPoints[i].Z;
        double vx = UEPoints[i].X;
        double vy = UEPoints[i].Y;
        double vz = UEPoints[i].Z;

        int row = 3 * i;
        // row for vx
        M(row, 0) = x; M(row, 1) = y; M(row, 2) = z; M(row, 3) = 1.0;
        for (int c = 4; c < 12; ++c) M(row, c) = 0.0;
        b(row) = vx;

        // row for vy
        M(row + 1, 0) = 0.0; M(row + 1, 1) = 0.0; M(row + 1, 2) = 0.0; M(row + 1, 3) = 0.0;
        M(row + 1, 4) = x; M(row + 1, 5) = y; M(row + 1, 6) = z; M(row + 1, 7) = 1.0;
        for (int c = 8; c < 12; ++c) M(row + 1, c) = 0.0;
        b(row + 1) = vy;

        // row for vz
        for (int c = 0; c < 8; ++c) M(row + 2, c) = 0.0;
        M(row + 2, 8) = x; M(row + 2, 9) = y; M(row + 2, 10) = z; M(row + 2, 11) = 1.0;
        b(row + 2) = vz;
    }

    // Solve least squares via normal equations or SVD (more stable).
    Eigen::VectorXd x = M.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    // x is size 12
    // Pack into OutMatrix3x4 row-major
    OutMatrix3x4.Empty();
    OutMatrix3x4.SetNum(12);
    for (int i = 0; i < 12; ++i) OutMatrix3x4[i] = static_cast<float>(x(i));

    if (bLog)
    {
        UE_LOG(LogTemp, Log, TEXT("Affine matrix (row-major 3x4):"));
        FString matLine;
        for (int r = 0; r < 3; ++r)
        {
            matLine.Empty();
            for (int c = 0; c < 4; ++c)
            {
                int idx = r * 4 + c;
                matLine += FString::Printf(TEXT("%f "), OutMatrix3x4[idx]);
            }
            UE_LOG(LogTemp, Log, TEXT("[%s]"), *matLine);
        }

        // Print residuals
        for (int i = 0; i < N; ++i)
        {
            // apply affine
            double xx = IRLPoints[i].X;
            double yy = IRLPoints[i].Y;
            double zz = IRLPoints[i].Z;
            Eigen::Vector3d pred;
            pred(0) = x(0) * xx + x(1) * yy + x(2) * zz + x(3);
            pred(1) = x(4) * xx + x(5) * yy + x(6) * zz + x(7);
            pred(2) = x(8) * xx + x(9) * yy + x(10) * zz + x(11);
            FVector PredF((float)pred(0), (float)pred(1), (float)pred(2));
            UE_LOG(LogTemp, Log, TEXT("Pt %d  IRL=%s  UE_expected=%s  Pred=%s  Delta=%s"),
                i, *IRLPoints[i].ToString(), *UEPoints[i].ToString(), *PredF.ToString(), *(PredF - UEPoints[i]).ToString());
        }
    }

    return true;
}

// Apply stored affine matrix (in AffineMatrix3x4) to a point. Exposed to Blueprint.
FVector UOpenVRManager::TransformPositionAffine(const FVector& InPosition, const TArray<float>& AffineMatrix3x4) const
{
    if (AffineMatrix3x4.Num() != 12)
    {
        UE_LOG(LogTemp, Warning, TEXT("Affine matrix not initialized"));
        return FVector::ZeroVector;
    }

    double x = InPosition.X;
    double y = InPosition.Y;
    double z = InPosition.Z;

    double px = AffineMatrix3x4[0] * x + AffineMatrix3x4[1] * y + AffineMatrix3x4[2] * z + AffineMatrix3x4[3];
    double py = AffineMatrix3x4[4] * x + AffineMatrix3x4[5] * y + AffineMatrix3x4[6] * z + AffineMatrix3x4[7];
    double pz = AffineMatrix3x4[8] * x + AffineMatrix3x4[9] * y + AffineMatrix3x4[10] * z + AffineMatrix3x4[11];

    return FVector((float)px, (float)py, (float)pz);
}
