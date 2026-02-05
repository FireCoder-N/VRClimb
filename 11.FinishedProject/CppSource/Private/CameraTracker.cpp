// Fill out your copyright notice in the Description page of Project Settings.


#include "CameraTracker.h"


// Sets default values
ACameraTracker::ACameraTracker()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void ACameraTracker::BeginPlay()
{
	Super::BeginPlay();

    VRManager = NewObject<UOpenVRManager>();
    //VRManager->Initialize();

    //Capture->openCamera();
    //Capture->captureDepthImage("C:/Users/Mike/Documents/9.Scene/MyProject/captures/");
    //Capture->~ZedDepthCapture();
	
}

// Called every frame
void ACameraTracker::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void ACameraTracker::Activate()
{
    Capture->openCamera();
    Capture->captureDepthImage("C:/Users/Mike/Documents/9.Scene/MyProject/captures/");
    Capture->~ZedDepthCapture();

    if (VRManager)
    {
        //VRManager->UpdateTracking();
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("VR tracking failed"));
    }
}