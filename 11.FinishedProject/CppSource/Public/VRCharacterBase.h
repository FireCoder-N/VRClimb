// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "CineCameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "VRCharacterBase.generated.h"

UCLASS()
class MYPROJECT_API AVRCharacterBase : public ACharacter
{
	GENERATED_BODY()

public:
	// Sets default values for this character's properties
	AVRCharacterBase();

	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	/*UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	USceneComponent* VROffset;*/

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UCineCameraComponent* VRCamera;


};
