// Fill out your copyright notice in the Description page of Project Settings.

#include "MoonController.h"
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

// Sets default values for this component's properties
UMoonController::UMoonController(){
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;
}

void UMoonController::SetParameters(
	float MoonBrightnessIn,
	float MoonSizeIn,
	FVector MoonColorIn,
    float GlowBrightnessIn,
	float GlowSizeIn,
	FVector GlowColorIn,
    float GlowSharpnessIn,
    int CycleIndexIn,
    int CycleDaysIn
	){

	DefaultMoonBrightness = MoonBrightnessIn;
	DefaultMoonSize = MoonSizeIn;
	DefaultMoonColor = MoonColorIn;
	DefaultGlowBrightness = GlowBrightnessIn;
    DefaultGlowSize = GlowSizeIn;
	DefaultGlowColor = GlowColorIn;
    DefaultGlowSharpness = GlowSharpnessIn;
    CycleDays = CycleDaysIn;
    CycleIndex = CycleIndexIn;

    UE_LOG(LogTemp, Warning, TEXT("mb: %f, ms: %f, gs: %f, gb: %f"), DefaultMoonBrightness, DefaultMoonSize, DefaultGlowSize, DefaultGlowBrightness);
}

void UMoonController::UpdateCycleIndex()
{
	CycleIndex = (CycleIndex + 1) % CycleDays;
	UpdateMoonParametersByCycleDay();
}

void UMoonController::UpdateMoonParametersByCycleDay(){
    const float FullMoonDay = CycleDays / 2; // Middle of the cycle is full moon
	const FVector FullMoonColor(1.0f, 1.0f, 1.0f); // White
	const FVector NewMoonColor(0.82f,0.82f,0.9f); // Grayish blue

    // Compute illumination percentage based on the day of the cycle
    float Illumination = FMath::Abs(CycleIndex - FullMoonDay) / FullMoonDay;
    Illumination = 1.0f - Illumination; // Invert for brightness (peaks at full moon)
	
	// ---------------------------------------------------------
    // Calculate parameters with randomness and relative scaling
	// ---------------------------------------------------------
	float RandomVariation;

	// Brightness
	RandomVariation = FMath::RandRange(-0.05f, 0.05f);
    CycleBrightnessFactor = FMath::Lerp(MinBrightness, MaxBrightness, Illumination + RandomVariation);
    /*GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, FString::Printf(TEXT("Illumination: %f, CycleBrightness: %f"), Illumination, CycleBrightnessFactor));
    UE_LOG(LogTemp, Warning, TEXT("Illumination: %f, CycleBrightness: %f"), Illumination, CycleBrightnessFactor);*/

	// Size (apogee and perigee variations)
    float Orbital = FMath::Sin((CycleIndex / CycleDays) * 2 * PI);		 // Sine-based orbit
    CycleSizeFactor = (1.0f + SizeVariation * Orbital);

	// Color
	RandomVariation = FMath::RandRange(-0.01f, 0.01f);
	CycleColorFactor = FMath::Lerp(NewMoonColor, FullMoonColor, Illumination + RandomVariation);


	// Glow Size
	RandomVariation = FMath::RandRange(-0.05f, 0.05f);
    CycleGlowSizeFactor = FMath::Lerp(MinGlowSize, MaxGlowSize, Illumination) + RandomVariation;
    
    // Glow brightness
    CycleGlowBrightnessFactor = FMath::Lerp(MinGlowBrightness, MaxGlowBrightness, Illumination);

	// Glow Color
	CycleGlowColorFactor = CycleColorFactor * CycleGlowBrightnessFactor;

	// Glow Sharpness Modifier
	CycleGlowSharpnessFactor = DefaultGlowSharpness - FMath::Abs(2.0f * Illumination - 1.0f); //FMath::RandRange(-0.1f, 0.1f);
}

void UMoonController::UpdateMoonParametersByMonth(int MonthIndex)
{
    const FVector WinterColor(0.95f, 0.95f, 1.0f); // Cool bluish-white
    const FVector SummerColor(1.0f, 0.98f, 0.95f); // Warm yellowish-white

    // Determine the normalized month index (1–12 -> 0.0–1.0)
    float NormalizedMonth = (MonthIndex - 1) / 11.0f; // MonthIndex: 1=Jan, 12=Dec

    // ---------------------------------------------------------
    // Calculate seasonal parameters with randomness and scaling
    // ---------------------------------------------------------

    float RandomVariation;

    // Brightness (dimmer in summer, brighter in winter)
    RandomVariation = FMath::RandRange(-0.05f, 0.05f);
    MonthBrightnessFactor = FMath::Lerp(MaxBrightness, MinBrightness, NormalizedMonth) + RandomVariation;

    // Size (slight variation between summer and winter months)
    RandomVariation = FMath::RandRange(-0.01f, 0.01f);
    MonthSizeFactor = 1.0f + FMath::Lerp(-SizeVariation, SizeVariation, NormalizedMonth) + RandomVariation;

    // Color (warmer in summer, cooler in winter)
    RandomVariation = FMath::RandRange(-0.02f, 0.02f);
    MonthColorFactor = FMath::Lerp(WinterColor, SummerColor, NormalizedMonth) + FVector(RandomVariation);

    // Glow Size (larger glow in winter for a more prominent appearance)
    RandomVariation = FMath::RandRange(-0.05f, 0.05f);
    MonthGlowSizeFactor = FMath::Lerp(MaxGlowSize, MinGlowSize, NormalizedMonth) + RandomVariation;

    // Glow Brightness (dimmer glow in summer months)
    RandomVariation = FMath::RandRange(-0.03f, 0.03f);
    MonthGlowBrightnessFactor = FMath::Lerp(MaxGlowBrightness, MinGlowBrightness, NormalizedMonth) + RandomVariation;

    // Glow Color (matches the seasonal color)
    MonthGlowColorFactor = MonthColorFactor * MonthGlowBrightnessFactor;

    // // Glow Sharpness Modifier (slightly sharper in summer for a smaller glow)
    // MonthGlowSharpnessFactor = 1.0f - NormalizedMonth + FMath::RandRange(-0.05f, 0.05f);
}

void UMoonController::UpdateMoonParametersByTimeOfNight(float TimeOfNight)
{
    // Normalize time (assuming TimeOfNight is between 0.0 and 1.0, where 0.0 = sunset, 1.0 = sunrise)
    float NightProgress = FMath::Clamp(TimeOfNight, 0.0f, 1.0f);

    // **Brightness Factor**
    TimeBrightnessFactor = FMath::InterpEaseInOut(0.4f, 1.2f, NightProgress, 2.5f);
    // *Realistic*: FMath::InterpEaseInOut(0.8f, 1.0f, NightProgress, 2.0f);

    // **Size Factor**
    // *Realistic*: 1.0f + 0.02f * FMath::Sin(NightProgress * PI);
    TimeSizeFactor = 1.0f + 0.05f * FMath::Sin(NightProgress * PI);

    // **Color Factor**
    FVector HorizonColor(1.0f, 0.6f, 0.4f); // Warm orange at horizon
    FVector PeakColor(1.0f, 1.0f, 1.0f);   // White at zenith
    TimeColorFactor = FMath::Lerp(HorizonColor, PeakColor, NightProgress);
    // *Realistic*: Less intense gradient

    // **Glow Size Factor**
    TimeGlowSizeFactor = FMath::Lerp(1.2f, 0.8f, NightProgress);

    // **Glow Brightness Factor**
    TimeGlowBrightnessFactor = FMath::InterpEaseInOut(0.7f, 1.3f, NightProgress, 2.0f);
    // *Realistic*: FMath::InterpEaseInOut(0.8f, 1.2f, NightProgress, 2.0f);

    // **Glow Color Factor**
    TimeGlowColorFactor = TimeColorFactor * TimeGlowBrightnessFactor;

    // // **Glow Sharpness Factor**
    // // - Realistic: Soft glow at dusk/dawn, sharper at peak
    // // - Exaggerated: Super sharp at midnight, super soft at dusk/dawn
    // TimeGlowSharpnessFactor = 1.0f - FMath::Abs(2.0f * NightProgress - 1.0f) + FMath::RandRange(-0.1f, 0.1f);
    // // *Realistic*: 1.0f - FMath::Abs(2.0f * NightProgress - 1.0f);
}




void UMoonController::GetMoonParameters(
	int& Upos,
	int& Vpos,
	float& MoonBrightnessFactor,
	float& MoonSizeFactor,
	FVector& MoonColorFactor,
	float& GlowSizeFactor,
	float& GlowBrightnessFactor,
	FVector& GlowColorFactor,
	float& GlowSharpnessFactor
	)
{
	int TextureIndex = (int) ceil(CycleIndex / 2.0f);
	Upos = (int)TextureIndex % 4;
	Vpos = (int)TextureIndex / 4;

	//MoonBrightnessFactor = clamp(DefaultMoonBrightness * CycleBrightnessFactor * MonthBrightnessFactor * TimeBrightnessFactor, MinBrightness, MaxBrightness);
    MoonBrightnessFactor = DefaultMoonBrightness * CycleBrightnessFactor * MonthBrightnessFactor * TimeBrightnessFactor;
    MoonSizeFactor = DefaultMoonSize * CycleSizeFactor * MonthSizeFactor * TimeSizeFactor;
	MoonColorFactor = DefaultMoonColor * CycleColorFactor * MonthColorFactor * TimeColorFactor;

    GlowSizeFactor = DefaultGlowSize * CycleGlowSizeFactor * MonthGlowSizeFactor * TimeGlowSizeFactor; //, MinGlowSize, MaxGlowSize; // * MonthGlowSizeFactor * TimeGlowSizeFactor;
    GlowBrightnessFactor = DefaultGlowBrightness * CycleGlowBrightnessFactor * MonthGlowBrightnessFactor * TimeGlowBrightnessFactor; // , MinGlowBrightness, MaxGlowBrightness); //  * TimeGlowBrightnessFactor;
    
	GlowColorFactor = DefaultGlowColor * CycleGlowColorFactor * MonthGlowColorFactor * TimeGlowColorFactor;
	GlowSharpnessFactor = CycleGlowSharpnessFactor;

    //UE_LOG(LogTemp, Warning, TEXT("Moon Params: (%d, %d), b: %f, gb: %f"), Upos, Vpos, MoonBrightnessFactor, GlowBrightnessFactor);
    //GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, FString::Printf(TEXT("Default: %f, Cycle: %f, Month: %f, Time: %f, Total: %f"), DefaultGlowSize, CycleGlowSizeFactor, MonthGlowSizeFactor, TimeGlowSizeFactor, GlowSizeFactor));
}


float UMoonController::GetNormalizedNightTime(float SunriseTime, float SunsetTime, float SolarTime)
{
    // Ensure times are valid
    if (SunsetTime < SunriseTime)
    {
        return 0.0f; // Invalid case, night cannot be computed
    }

    // Determine if it's currently nighttime
    if (SolarTime > SunsetTime || SolarTime < SunriseTime)
    {
        // Normalize the time into the 0.0 - 1.0 range
        float NightDuration = (24.0f - SunsetTime) + SunriseTime; // Total night hours, accounting for midnight
        float NightProgress;

        if (SolarTime >= SunsetTime) // Evening hours (before midnight)
        {
            NightProgress = (SolarTime - SunsetTime) / NightDuration;
        }
        else // Early morning hours (after midnight)
        {
            NightProgress = (SolarTime + (24.0f - SunsetTime)) / NightDuration;
        }

        return FMath::Clamp(NightProgress, 0.0f, 1.0f);
    }

    return 0.0f; // If it's daytime, return 0 (not nighttime)
}
