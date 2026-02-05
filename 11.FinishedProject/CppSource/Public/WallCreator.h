// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "ProceduralMeshComponent.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/Filehelper.h"
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "WallCreator.generated.h"



//class UProceduralMeshComponent;

UCLASS()
class MYPROJECT_API AWallCreator : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AWallCreator();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	UPROPERTY(EditAnywhere) //, Meta = (MakeEditWidget = true))
	TArray<FVector> Vertices;

	UPROPERTY(EditAnywhere)
	TArray<int> Triangles;

	UPROPERTY(EditAnywhere)
	UMaterialInterface* Material;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UFUNCTION(BlueprintCallable, Category = "Meshify")
	bool ReadOBJFile(FString FilePath);

private:
	UProceduralMeshComponent* ProceduralMesh;

};
