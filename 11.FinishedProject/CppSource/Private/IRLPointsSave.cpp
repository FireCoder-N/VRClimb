// Fill out your copyright notice in the Description page of Project Settings.


#include "IRLPointsSave.h"
#include "Kismet/GameplayStatics.h"


void UIRLPointsSave::SavePoints(const TArray<FVector>& Points)
{
	UIRLPointsSave* SaveInstance = Cast<UIRLPointsSave>(
		UGameplayStatics::CreateSaveGameObject(UIRLPointsSave::StaticClass())
	);

	SaveInstance->SavedIRLPoints = Points;

	UGameplayStatics::SaveGameToSlot(SaveInstance, TEXT("IRLPointsSlot"), 0);
}

TArray<FVector> UIRLPointsSave::LoadPoints()
{
	if (UGameplayStatics::DoesSaveGameExist(TEXT("IRLPointsSlot"), 0)) {
		UIRLPointsSave* LoadedGame = Cast<UIRLPointsSave>(
			UGameplayStatics::LoadGameFromSlot(TEXT("IRLPointsSlot"), 0)
		);

		if (LoadedGame) {
			return LoadedGame->SavedIRLPoints;
		}
	}

	return {};
}