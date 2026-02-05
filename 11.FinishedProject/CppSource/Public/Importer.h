// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Importer.generated.h"

class UAssetImportTask;
class UFactory;

/**
 * 
 */
UCLASS()
class MYPROJECT_API UImporter : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:

	UFUNCTION(BlueprintCallable, Category="Importer")
	static UAssetImportTask* CreateImportTask(
		FString SourcePath, 
		FString DestinationPath, 
		UFactory* ExtraFactory, 
		UObject* ExtraOptions, 
		bool& bOutSuccess,
		FString& OutInfoMessage);

	UFUNCTION(BlueprintCallable, Category = "Importer")
	static UObject* ProcessImportTask(UAssetImportTask* ImportTask, bool& bOutSuccess, FString& OutInfoMessage);

	UFUNCTION(BlueprintCallable, Category = "Importer")
	static void GetFourCorners(UStaticMeshComponent* MeshComponent, TArray<FVector>& OutCorners);
};
