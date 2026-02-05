// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "Components/InputComponent.h"
#include "GameFramework/PlayerController.h"
#include "openvr.h"
#include "OpenVRManager.h"
#include "ZedDepthCapture.h"
#include "CameraTracker.generated.h"

UCLASS()
class MYPROJECT_API ACameraTracker : public APawn
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	ACameraTracker();

	// Called every frame
	virtual void Tick(float DeltaTime) override;
	

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	UOpenVRManager* VRManager;
	ZedDepthCapture* Capture = new ZedDepthCapture();

	UFUNCTION(BlueprintCallable, Category = "Camera Actirons")
	void Activate();
};