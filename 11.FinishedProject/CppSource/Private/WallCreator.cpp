// Fill out your copyright notice in the Description page of Project Settings.
#include "WallCreator.h"

// Sets default values
AWallCreator::AWallCreator()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;

	ProceduralMesh = CreateDefaultSubobject<UProceduralMeshComponent>("ProceduralMesh");
	ProceduralMesh->SetupAttachment(GetRootComponent());

}

// Called when the game starts or when spawned
void AWallCreator::BeginPlay()
{
	Super::BeginPlay();

    /*ReadOBJFile("C:/Users/Mike/Documents/9.Scene/MyProject/Scripts/output.obj");*/
	
	ProceduralMesh->CreateMeshSection(0, Vertices, Triangles, TArray<FVector>(), TArray<FVector2D>(), TArray<FColor>(), TArray<FProcMeshTangent>(), true);
	ProceduralMesh->SetMaterial(0, Material);
}

// Called every frame
void AWallCreator::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}


bool AWallCreator::ReadOBJFile(FString FilePath)
{
    if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*FilePath))
    {
		UE_LOG(LogTemp, Error, TEXT("File does not exist %s"), *FilePath);
        return false;
    }

	TArray<FString> FileLines;
	if (FFileHelper::LoadFileToStringArray(FileLines, *FilePath))
	{
		// Temporary variables to store parsed data
		TArray<FVector> TempVertices;
		TArray<int> TempTriangles;

		// Parse each line in the OBJ file
		for (const FString& Line : FileLines)
		{
			// Ignore empty lines and comments
			if (Line.IsEmpty() || Line.StartsWith(TEXT("#"))) continue;

			// Parse vertex lines (v x y z)
			if (Line.StartsWith(TEXT("v ")))
			{
				// Split the line into tokens based on space
				TArray<FString> Tokens;
				Line.ParseIntoArray(Tokens, TEXT(" "), true);

				// Convert tokens to float values and create a FVector
				if (Tokens.Num() >= 4)
				{
					float X = FCString::Atof(*Tokens[1]);
					float Y = FCString::Atof(*Tokens[2]);
					float Z = FCString::Atof(*Tokens[3]);
					TempVertices.Add(FVector(X, Y, Z));
				}
			}
			// Parse face lines (f vertex1/uv1/normal1 vertex2/uv2/normal2 ...)
			else if (Line.StartsWith(TEXT("f ")))
			{
				// Split the face data by spaces
				TArray<FString> FaceTokens;
				Line.ParseIntoArray(FaceTokens, TEXT(" "), true);

				// Extract the vertex indices (ignoring UV and normal indices)
				for (const FString& FaceToken : FaceTokens)
				{
					TArray<FString> Indices;
					FaceToken.ParseIntoArray(Indices, TEXT("/"), true);
					if (Indices.Num() > 0)
					{
						// OBJ indices are 1-based, so we subtract 1 to convert to 0-based indices
						int32 VertexIndex = FCString::Atoi(*Indices[0]) - 1;
						TempTriangles.Add(VertexIndex);
					}
				}
				TempTriangles.RemoveAll([](int32 Index) {
					return Index < 0;
				});
			}
		}
		// Assign parsed data to your class's properties
		Vertices = TempVertices;
		Triangles = TempTriangles;
		return true;
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load OBJ file at %s"), *FilePath);
		return false;
	}
}

