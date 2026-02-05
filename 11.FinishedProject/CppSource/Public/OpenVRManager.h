// OpenVRManager.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "openvr.h"  // Include the OpenVR header
#include "Kismet/KismetMathLibrary.h"
#include "Eigen/Dense"
#include "OpenVRManager.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class MYPROJECT_API UOpenVRManager : public UActorComponent
{
    GENERATED_BODY()

public:
    // Initializes OpenVR; returns true on success.
    UFUNCTION(BlueprintCallable, Category = "OpenVR")
    bool Initialize();

    // Poll tracking data (call this every Tick).
    UFUNCTION(BlueprintCallable, Category = "OpenVR")
    FString DebugTracking();

    // Shutdown OpenVR.
    UFUNCTION(BlueprintCallable, Category = "OpenVR")
    void Shutdown();

    UFUNCTION(BlueprintCallable, Category = "OpenVR")
    void GetTrackerPosition(int index, float& x, float& y, float& z);

    UFUNCTION(BlueprintCallable, Category = "OpenVR")
    void GetTrackerOrientation(int index, FQuat& Q);

    UFUNCTION(BlueprintCallable, Category = "OpenVR")
    static bool ComputeCalibrationTransform(
        const TArray<FVector>& IRLPoints,
        const TArray<FVector>& UEPoints,
        FTransform& OutTransform
    );

    UFUNCTION(BlueprintCallable, Category = "OpenVR")
    void CalibrateTrackers(TArray<FVector>& IrlPoints);

    UFUNCTION(BlueprintCallable, Category = "Calibration")
    static bool ComputeAffineTransform(
        const TArray<FVector>& IRLPoints,
        const TArray<FVector>& UEPoints,
        TArray<float>& OutMatrix3x4,
        bool bLog = false);

    UFUNCTION(BlueprintCallable, Category = "Calibration")
    FVector TransformPositionAffine(const FVector& InPosition, const TArray<float>& AffineMatrix3x4) const;

    //// store matrix in class for runtime use:
    //UPROPERTY(BlueprintReadWrite, Category = "Calibration")
    //TArray<float> AffineMatrix3x4; // row-major 3x4

private:
    vr::IVRSystem* VRSystem = nullptr;

    // Helper function to extract position from a 3x4 matrix.
    void GetPosition(const vr::HmdMatrix34_t& mat, float& x, float& y, float& z);

    void GetOrientation(const vr::HmdMatrix34_t& mat, FQuat& R);

    void GetOrientationUnreal(const vr::HmdMatrix34_t& mat, FQuat& Q);

    void GetOrientationUnreal_HMD(const vr::HmdMatrix34_t& mat, FQuat& Q);

public:
    UPROPERTY(EditAnywhere, Category = "Tracker Data")
    float HMD_Offset = 0.0;

};
