// Fill out your copyright notice in the Description page of Project Settings.


#include "Importer.h"

#include "Editor/UnrealEd/Public/AssetImportTask.h"
#include "AssetToolsModule.h"

UAssetImportTask* UImporter::CreateImportTask(FString SourcePath, FString DestinationPath, UFactory* ExtraFactory, UObject* ExtraOptions,
	bool& bOutSuccess, FString& OutInfoMessage) {

	UAssetImportTask* RetTask = NewObject<UAssetImportTask>();

	if (RetTask == nullptr) {
		bOutSuccess = false;
		UE_LOG(LogTemp, Error, TEXT("Create Import Task failed for %s"), *SourcePath);
		return nullptr;
	}

	RetTask->Filename = SourcePath;
	RetTask->DestinationPath = FPaths::GetPath(DestinationPath);
	RetTask->DestinationName = FPaths::GetCleanFilename(DestinationPath);

	RetTask->bSave = false;
	RetTask->bAutomated = true;
	RetTask->bAsync = false;
	RetTask->bReplaceExisting = true;
	RetTask->bReplaceExistingSettings = false;

	RetTask->Factory = ExtraFactory;
	RetTask->Options = ExtraOptions;

	bOutSuccess = true;
	OutInfoMessage = FString::Printf(TEXT("Create Import Task Succeeded"));
	return RetTask;
}


UObject* UImporter::ProcessImportTask(UAssetImportTask* ImportTask, bool& bOutSuccess, FString& OutInfoMessage) {
	if (ImportTask == nullptr) {
		bOutSuccess = false;
		UE_LOG(LogTemp, Error, TEXT("Process Import Task failed."));
		return nullptr;
	}

	FAssetToolsModule* AssetTools = FModuleManager::LoadModulePtr<FAssetToolsModule>("AssetTools");

	if (AssetTools == nullptr) {
		bOutSuccess = false;
		UE_LOG(LogTemp, Error, TEXT("AssetTools Module is invalid"));
		return nullptr;
	}

	AssetTools->Get().ImportAssetTasks({ ImportTask });

	if (ImportTask->GetObjects().Num() == 0) {
		bOutSuccess = false;
		UE_LOG(LogTemp, Error, TEXT("Nothing was Imported. Asset type suppoerted? %s"), *ImportTask->Filename);
		return nullptr;
	}

	UObject* ImportedAsset = StaticLoadObject(UObject::StaticClass(), nullptr, *FPaths::Combine(ImportTask->DestinationPath, ImportTask->DestinationName));

	bOutSuccess = true;
	OutInfoMessage = FString::Printf(TEXT("Import of %s Succeeded"), *ImportTask->Filename);
	return ImportedAsset;
}

void UImporter::GetFourCorners(UStaticMeshComponent* MeshComponent, TArray<FVector>& OutCorners) {
	if (!MeshComponent || !MeshComponent->GetStaticMesh()) {
		UE_LOG(LogTemp, Warning, TEXT("Invalid Mesh Component"));
		return;
	}

	FBox LocalBounds = MeshComponent->GetStaticMesh()->GetBoundingBox();
	FVector Min = LocalBounds.Min;
	FVector Max = LocalBounds.Max;

	FVector BottomLeft;
	FVector BottomRight;
	FVector TopLeft;
	FVector TopRight;

	FVector Extent = LocalBounds.GetSize();
	if (Extent.X < Extent.Y && Extent.X < Extent.Z)
	{
		// X is the thin axis — use YZ plane
		UE_LOG(LogTemp, Warning, TEXT("YZ plane"));
		float X = 0.0f;
		BottomLeft = FVector(X, Min.Y, Min.Z);
		BottomRight = FVector(X, Max.Y, Min.Z);
		TopRight = FVector(X, Max.Y, Max.Z);
		TopLeft = FVector(X, Min.Y, Max.Z);
	}
	else if (Extent.Y < Extent.X && Extent.Y < Extent.Z)
	{
		// Y is the thin axis — use XZ plane
		UE_LOG(LogTemp, Warning, TEXT("XZ plane"));
		float Y = 0.0f;
		BottomLeft = FVector(Min.X, Y, Min.Z);
		BottomRight = FVector(Max.X, Y, Min.Z);
		TopRight = FVector(Max.X, Y, Max.Z);
		TopLeft = FVector(Min.X, Y, Max.Z);
	}
	else
	{
		// Z is the thin axis — use XY plane (default)
		UE_LOG(LogTemp, Warning, TEXT("XY plane"));
		float Z = 0.0f;
		BottomLeft = FVector(Min.X, Min.Y, Z);
		BottomRight = FVector(Max.X, Min.Y, Z);
		TopRight = FVector(Max.X, Max.Y, Z);
		TopLeft = FVector(Min.X, Max.Y, Z);
	}

	// Convert to world space if needed
	BottomLeft = MeshComponent->GetComponentTransform().TransformPosition(BottomLeft);
	BottomRight = MeshComponent->GetComponentTransform().TransformPosition(BottomRight);
	TopRight = MeshComponent->GetComponentTransform().TransformPosition(TopRight);
	TopLeft = MeshComponent->GetComponentTransform().TransformPosition(TopLeft);

	// Store in output array
	OutCorners = { BottomLeft, BottomRight, TopRight, TopLeft };
}