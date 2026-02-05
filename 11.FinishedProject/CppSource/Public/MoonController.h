// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MoonController.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class MYPROJECT_API UMoonController : public UActorComponent
{
	GENERATED_BODY()

private:
	// Default parameters
	float DefaultMoonBrightness, DefaultMoonSize;
	FVector DefaultMoonColor;
	float DefaultGlowBrightness, DefaultGlowSize, DefaultGlowSharpness;
	FVector DefaultGlowColor;

	int CycleIndex;
	int CycleDays;

    // --- Scaling factors calculated by individual functions ---

	// Day of the cycle
    float CycleBrightnessFactor = 1.0f, CycleSizeFactor = 1.0f;
    FVector CycleColorFactor = FVector(1.0f, 1.0f, 1.0f);
    float CycleGlowBrightnessFactor = 1.0f, CycleGlowSizeFactor = 1.0f;
    FVector CycleGlowColorFactor = FVector(1.0f, 1.0f, 1.0f);
	float CycleGlowSharpnessFactor = 1.0f;

	// Month of the year
	float MonthBrightnessFactor = 1.0f, MonthSizeFactor = 1.0f;
	FVector MonthColorFactor = FVector(1.0f, 1.0f, 1.0f);
	float MonthGlowBrightnessFactor = 1.0f, MonthGlowSizeFactor = 1.0f;
	FVector MonthGlowColorFactor = FVector(1.0f, 1.0f, 1.0f);

	// Time of the day
	float TimeBrightnessFactor = 1.0f, TimeSizeFactor = 1.0f;
	FVector TimeColorFactor = FVector(1.0f, 1.0f, 1.0f);
	float TimeGlowBrightnessFactor = 1.0f, TimeGlowSizeFactor = 1.0f;
	FVector TimeGlowColorFactor = FVector(1.0f, 1.0f, 1.0f);


public:
	//Cycle settings
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle Settings")
    float MaxBrightness = 1.0f; // Full moon brightness

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle Settings")
    float MinBrightness = 0.0f; // New moon brightness

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle Settings")
	float SizeVariation = 0.14f; // 14% size change


	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle Settings")
	float MaxGlowSize = 0.8f;  // Max glow size multiplier

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle Settings")
    float MinGlowSize = 0.1f;  // Min glow size multiplier

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle Settings")
	float MaxGlowBrightness = 0.9f;  // Max glow brightness (full moon)

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle Settings")
    float MinGlowBrightness = 0.03f;  // Min glow brightness (new moon)





public:
	UMoonController();

	UFUNCTION(BlueprintCallable, Category = "Moon")
	void SetParameters(
		float MoonBrightnessIn,
		float MoonSizeIn,
		FVector MoonColorIn,
		float GlowBrightnessIn,
		float GlowSizeIn,
		FVector GlowColorIn,
		float GlowSharpnessIn,
		int CycleIndexIn,
		int CycleDaysIn
	);

	UFUNCTION(BlueprintCallable, Category = "Moon")
	void UpdateCycleIndex();

	UFUNCTION(BlueprintCallable, Category = "Moon")
	void UpdateMoonParametersByCycleDay();

	UFUNCTION(BlueprintCallable, Category = "Moon")
	void UpdateMoonParametersByMonth(int MonthIndex);

	UFUNCTION(BlueprintCallable, Category = "Moon")
	void UpdateMoonParametersByTimeOfNight(float TimeOfNight);


	UFUNCTION(BlueprintCallable, Category = "Moon")
	void GetMoonParameters(
		int& Upos,
		int& Vpos,
		float& MoonBrightnessFactor,
		float& MoonSizeFactor,
		FVector& MoonColorFactor,
		float& GlowBrightnessFactor,
		float& GlowSizeFactor,
		FVector& GlowColorFactor,
		float& GlowSharpnessFactor
	);

	UFUNCTION(BlueprintCallable, Category = "Moon")
	float GetNormalizedNightTime(float SunriseTime, float SunsetTime, float SolarTime);
};
