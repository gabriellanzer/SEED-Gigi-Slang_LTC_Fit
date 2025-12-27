// SlangAutoDiff technique, shader Render
/*$(ShaderResources)*/

#include "LTC_Math.slang"

// LTC Buffer Layout
static const int kLTC_STRIDE = 48;
static const int kIDx_LTC_Magnitude = 0;
static const int kIDx_LTC_Fresnel = 1;
static const int kIDx_LTC_m11 = 2;
static const int kIDx_LTC_m22 = 3;
static const int kIDx_LTC_m13 = 4;
static const int kIDx_LTC_X_0 = 5;
static const int kIDx_LTC_X_1 = 6;
static const int kIDx_LTC_X_2 = 7;
static const int kIDx_LTC_Y_0 = 8;
static const int kIDx_LTC_Y_1 = 9;
static const int kIDx_LTC_Y_2 = 10;
static const int kIDx_LTC_Z_0 = 11;
static const int kIDx_LTC_Z_1 = 12;
static const int kIDx_LTC_Z_2 = 13;
static const int kIDx_LTC_M_00 = 14;
static const int kIDx_LTC_M_01 = 15;
static const int kIDx_LTC_M_02 = 16;
static const int kIDx_LTC_M_10 = 17;
static const int kIDx_LTC_M_11 = 18;
static const int kIDx_LTC_M_12 = 19;
static const int kIDx_LTC_M_20 = 20;
static const int kIDx_LTC_M_21 = 21;
static const int kIDx_LTC_M_22 = 22;
static const int kIDx_LTC_invM_00 = 23;
static const int kIDx_LTC_invM_01 = 24;
static const int kIDx_LTC_invM_02 = 25;
static const int kIDx_LTC_invM_10 = 26;
static const int kIDx_LTC_invM_11 = 27;
static const int kIDx_LTC_invM_12 = 28;
static const int kIDx_LTC_invM_20 = 29;
static const int kIDx_LTC_invM_21 = 30;
static const int kIDx_LTC_invM_22 = 31;
static const int kIDx_LTC_detM = 32;
static const int kIDx_LTC_isIsotropic = 33;
static const int kIDx_LTC_Error = 34;

float3 LinearToSRGB(float3 linearCol)
{
    float3 sRGBLo = linearCol * 12.92;
    float3 sRGBHi = (pow(abs(linearCol), float3(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4)) * 1.055) - 0.055;
    float3 sRGB;
    sRGB.r = linearCol.r <= 0.0031308 ? sRGBLo.r : sRGBHi.r;
    sRGB.g = linearCol.g <= 0.0031308 ? sRGBLo.g : sRGBHi.g;
    sRGB.b = linearCol.b <= 0.0031308 ? sRGBLo.b : sRGBHi.b;
    return sRGB;
}

[shader("compute")]
/*$(_compute:csmain)*/(uint3 DTid : SV_DispatchThreadID)
{
	uint2 dims;
	Output.GetDimensions(dims.x, dims.y);
	uint2 px = DTid.xy;
    if (px.x >= dims.x || px.y >= dims.y) return;

	float2 uv = (float2(px) + 0.5f) / float2(dims);

    // Target
    float targetRoughness = /*$(Variable:TargetRoughness)*/;
    float targetTheta = /*$(Variable:TargetTheta)*/;
	// Clamp theta to [-pi/2, pi/2]
    targetTheta = clamp(-targetTheta, -1.570796f, 1.570796f);

    float alpha = max(targetRoughness * targetRoughness, 0.00001f);
    int2 tableSize = /*$(Variable:TableSize)*/;

    // Calculate table indices
    // Inverse of: roughness = a / (tableSize.y - 1)
    int a = int(targetRoughness * float(tableSize.y - 1));
    a = clamp(a, 0, tableSize.y - 1);

    // Inverse of: x = t / (tableSize.x - 1), ct = 1 - x*x, theta = acos(ct)
    // ct = cos(theta), x = sqrt(1 - ct)
    float ct_target = cos(targetTheta);
    float x_target = sqrt(max(0.0f, 1.0f - ct_target)); // Ensure non-negative
    int t = int(round(x_target * float(tableSize.x - 1)));
    t = clamp(t, 0, tableSize.x - 1);

    int index = a * tableSize.x + t;

    // Load Params
    int baseIdx = index * kLTC_STRIDE;
    
    LTC ltc;
    ltc.magnitude = LTCBuffer[baseIdx + kIDx_LTC_Magnitude];
    ltc.fresnel = LTCBuffer[baseIdx + kIDx_LTC_Fresnel];
    ltc.X = float3(LTCBuffer[baseIdx + kIDx_LTC_X_0], LTCBuffer[baseIdx + kIDx_LTC_X_1], LTCBuffer[baseIdx + kIDx_LTC_X_2]);
    ltc.Y = float3(LTCBuffer[baseIdx + kIDx_LTC_Y_0], LTCBuffer[baseIdx + kIDx_LTC_Y_1], LTCBuffer[baseIdx + kIDx_LTC_Y_2]);
    ltc.Z = float3(LTCBuffer[baseIdx + kIDx_LTC_Z_0], LTCBuffer[baseIdx + kIDx_LTC_Z_1], LTCBuffer[baseIdx + kIDx_LTC_Z_2]);
    
    if (targetTheta < 0.0f)
    {
        ltc.X.x = -ltc.X.x;
        ltc.Y.x = -ltc.Y.x;
        ltc.Z.x = -ltc.Z.x;
    }

    ltc.m11 = LTCBuffer[baseIdx + kIDx_LTC_m11];
    ltc.m22 = LTCBuffer[baseIdx + kIDx_LTC_m22];
    ltc.m13 = LTCBuffer[baseIdx + kIDx_LTC_m13];
    
    // Initialize matrices to zero
    ltc.M = float3x3(0,0,0, 0,0,0, 0,0,0);
    ltc.invM = float3x3(0,0,0, 0,0,0, 0,0,0);
    ltc.detM = 0.0f;
    ltc.isIsotropic = LTCBuffer[baseIdx + kIDx_LTC_isIsotropic];
    
    LTC_update(ltc);
    
    float error = LTCBuffer[baseIdx + kIDx_LTC_Error];
    
    // For GGX Eval, we need V
    float sinT = sin(targetTheta);
    float cosT = cos(targetTheta);
    if (cosT < 1e-4f)
    {
        cosT = 1e-4f;
        float sinT_mag = sqrt(1.0f - cosT * cosT);
        sinT = (sinT >= 0.0f) ? sinT_mag : -sinT_mag;
    }
    float3 V = float3(sinT, 0.0f, cosT);

    // Visualization
    float3 color = float3(0.0f, 0.0f, 0.0f);

    // Split screen
    bool showLTC = uv.x > 0.5f;
	if (/*$(Variable:LTCPreviewOnLeftSide)*/)
        showLTC = !showLTC;

    // Map to disk
    float2 center = showLTC ? float2(0.75f, 0.5f) : float2(0.25f, 0.5f);
    float2 localUV = (uv - center) * float2(dims.x / float(dims.y), 1.0f) * 2.5f; // Aspect ratio correction

    float r2 = dot(localUV, localUV);
    if (r2 <= 1.0f)
    {
        float z = sqrt(1.0f - r2);
        float3 L = float3(localUV.x, localUV.y, z);

        float val = 0.0f;

        if (showLTC)
        {
            val = LTC_eval(ltc, L);
        }
        else
        {
			float pdf;
            val = GGX_eval(V, L, alpha, pdf);
        }

        color = float3(val, val, val);
    }

    // Error bar at bottom
    if (uv.y < 0.05f)
    {
        float err = error;
        float bar = err * dims.x; // Scale up
        if (uv.x < bar) color = float3(1.0f, 0.0f, 0.0f);
        else color = float3(0.2f, 0.2f, 0.2f);
    }
	else
	{
		// Separator
		if (abs(uv.x - 0.5f) < 0.002f) color = float3(1.0f, 1.0f, 1.0f);
	}

	Output[px] = float4(LinearToSRGB(color), 1.0f);
}

/*
Shader Resources:
	Buffer LTCBuffer (as SRV)
	Texture Output (as UAV)
*/
