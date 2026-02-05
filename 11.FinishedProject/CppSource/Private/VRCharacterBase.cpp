// Fill out your copyright notice in the Description page of Project Settings.


#include "VRCharacterBase.h"

// Sets default values
AVRCharacterBase::AVRCharacterBase()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	/*VROffset = CreateDefaultSubobject<USceneComponent>("VROffset");
	VROffset->SetupAttachment(GetCapsuleComponent());*/

	VRCamera = CreateDefaultSubobject<UCineCameraComponent>("VRCamera");
	//VRCamera->SetupAttachment(VROffset);
	//VRCamera->SetRelativeLocation(FVector(0, 0, -1*GetCapsuleComponent()->GetScaledCapsuleHalfHeight()));

}

// Called when the game starts or when spawned
void AVRCharacterBase::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AVRCharacterBase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Called to bind functionality to input
void AVRCharacterBase::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}

