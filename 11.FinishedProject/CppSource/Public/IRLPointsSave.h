// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/SaveGame.h"
#include "IRLPointsSave.generated.h"

/**
 * 
 */
UCLASS(BlueprintType)
class MYPROJECT_API UIRLPointsSave : public USaveGame
{
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, Category="SaveData")
	TArray<FVector> SavedIRLPoints;

	UFUNCTION(BlueprintCallable, Category = "SaveData")
	static void SavePoints(const TArray<FVector>& Points);

	UFUNCTION(BlueprintCallable, Category = "SaveData")
	static TArray<FVector> LoadPoints();
};
