// SlangAutoDiff technique, shader Init
/*$(ShaderResources)*/

#include "LTC_Math.slang"

// LTC Buffer Layout:
// 0: float magnitude;
// 1: float fresnel;
// 2: float m11
// 3: float m22
// 4: float m13;
// 5-7:  float3 X;
// 8-10:  float3 Y
// 11-13: float3 Z;
// 14-22: float3x3 M;
// 23-31: float3x3 invM;
// 32: float detM;
// 33: float isIsotropic;

static const int kLTC_STRIDE = 48; // round to 48 for better alignment
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

// Optimization State Layout
static const int kIDx_LTC_Error = 34;
static const int kIDx_LTC_Iter = 35;
static const int kIDx_LTC_Lion_m_0 = 36;
static const int kIDx_LTC_Lion_m_1 = 37;
static const int kIDx_LTC_Lion_m_2 = 38;
static const int kIDx_LTC_Adam_v_0 = 39;
static const int kIDx_LTC_Adam_v_1 = 40;
static const int kIDx_LTC_Adam_v_2 = 41;

void ComputeAvgTerms(float3 V, float alpha, out float norm, out float fresnel, out float3 averageDir)
{
    norm = 0.0f;
    fresnel = 0.0f;
    averageDir = float3(0, 0, 0);

    const int sampleCount = /*$(Variable:SampleCount)*/;
    for (int j = 0; j < sampleCount; ++j)
    {
        for (int i = 0; i < sampleCount; ++i)
        {
            float U1 = (float(i) + 0.5f) / float(sampleCount);
            float U2 = (float(j) + 0.5f) / float(sampleCount);

            float3 L = GGX_sample(V, alpha, U1, U2);

            float pdf;
            float eval = GGX_eval(V, L, alpha, pdf);

            if (pdf > 0.0f)
            {
                float weight = eval / pdf;
                float3 H = normalize(V + L);

                norm += weight;
                fresnel += weight * pow(1.0f - saturate(dot(V, H)), 5.0f);
                averageDir += weight * L;
            }
        }
    }

    norm /= float(sampleCount * sampleCount);
    fresnel /= float(sampleCount * sampleCount);

    averageDir.y = 0.0f;

    // Safe normalize
    averageDir = normalize(averageDir);
}

[shader("compute")]
/*$(_compute:csmain)*/(uint3 DTid : SV_DispatchThreadID)
{
    if (/*$(Variable:initialized)*/)
    {
        return;
    }

    int t = DTid.x;
    int a = DTid.y;
    int2 tableSize = /*$(Variable:TableSize)*/;

    if (t >= tableSize.x || a >= tableSize.y) return;

    int index = a * tableSize.x + t;

    // Target
    float roughness = float(a) / float(tableSize.y - 1);
    float alpha = max(roughness * roughness, 0.00001f);
    
    float x = float(t) / float(tableSize.x - 1);
    float ct = 1.0f - x*x;
    float theta = min(1.57f, acos(ct));
    
    float sinT = sin(theta);
    float cosT = max(cos(theta), 1e-4f); // Clamp cosT
    float3 V = float3(sinT, 0.0f, cosT);

    float mag = 1.0f;
    float fresnel = 0.0f;
    float3 averageDir = float3(0, 0, 1);
    ComputeAvgTerms(V, alpha, mag, fresnel, averageDir);

    // Initialize LTC Struct
    LTC ltc;
    ltc.magnitude = mag;
    ltc.fresnel = fresnel;

    // 1. first guess for the fit
    ltc.X = float3(1, 0, 0);
    ltc.Y = float3(0, 1, 0);
    ltc.Z = float3(0, 0, 1);

    ltc.m11 = 1.0f;
    ltc.m22 = 1.0f;
    ltc.m13 = 0.0f;

    if (t != 0)
    {
        float3 L = averageDir;
        float3 T1 = normalize(float3(L.z, 0, -L.x));
        float3 T2 = float3(0, 1, 0);
        ltc.X = T1;
        ltc.Y = T2;
        ltc.Z = L;

        ltc.isIsotropic = 0.0f;
    }
    else
    {
        ltc.isIsotropic = 1.0f;
    }

    LTC_update(ltc);

    // Store on LTC Buffer
    int baseIdx = index * kLTC_STRIDE;
    LTCBuffer[baseIdx + kIDx_LTC_Magnitude] = ltc.magnitude;
    LTCBuffer[baseIdx + kIDx_LTC_Fresnel] = ltc.fresnel;
    LTCBuffer[baseIdx + kIDx_LTC_m11] = ltc.m11;
    LTCBuffer[baseIdx + kIDx_LTC_m22] = ltc.m22;
    LTCBuffer[baseIdx + kIDx_LTC_m13] = ltc.m13;
    LTCBuffer[baseIdx + kIDx_LTC_X_0] = ltc.X.x;
    LTCBuffer[baseIdx + kIDx_LTC_X_1] = ltc.X.y;
    LTCBuffer[baseIdx + kIDx_LTC_X_2] = ltc.X.z;
    LTCBuffer[baseIdx + kIDx_LTC_Y_0] = ltc.Y.x;
    LTCBuffer[baseIdx + kIDx_LTC_Y_1] = ltc.Y.y;
    LTCBuffer[baseIdx + kIDx_LTC_Y_2] = ltc.Y.z;
    LTCBuffer[baseIdx + kIDx_LTC_Z_0] = ltc.Z.x;
    LTCBuffer[baseIdx + kIDx_LTC_Z_1] = ltc.Z.y;
    LTCBuffer[baseIdx + kIDx_LTC_Z_2] = ltc.Z.z;
    LTCBuffer[baseIdx + kIDx_LTC_M_00] = ltc.M[0][0];
    LTCBuffer[baseIdx + kIDx_LTC_M_01] = ltc.M[0][1];
    LTCBuffer[baseIdx + kIDx_LTC_M_02] = ltc.M[0][2];
    LTCBuffer[baseIdx + kIDx_LTC_M_10] = ltc.M[1][0];
    LTCBuffer[baseIdx + kIDx_LTC_M_11] = ltc.M[1][1];
    LTCBuffer[baseIdx + kIDx_LTC_M_12] = ltc.M[1][2];
    LTCBuffer[baseIdx + kIDx_LTC_M_20] = ltc.M[2][0];
    LTCBuffer[baseIdx + kIDx_LTC_M_21] = ltc.M[2][1];
    LTCBuffer[baseIdx + kIDx_LTC_M_22] = ltc.M[2][2];
    LTCBuffer[baseIdx + kIDx_LTC_invM_00] = ltc.invM[0][0];
    LTCBuffer[baseIdx + kIDx_LTC_invM_01] = ltc.invM[0][1];
    LTCBuffer[baseIdx + kIDx_LTC_invM_02] = ltc.invM[0][2];
    LTCBuffer[baseIdx + kIDx_LTC_invM_10] = ltc.invM[1][0];
    LTCBuffer[baseIdx + kIDx_LTC_invM_11] = ltc.invM[1][1];
    LTCBuffer[baseIdx + kIDx_LTC_invM_12] = ltc.invM[1][2];
    LTCBuffer[baseIdx + kIDx_LTC_invM_20] = ltc.invM[2][0];
    LTCBuffer[baseIdx + kIDx_LTC_invM_21] = ltc.invM[2][1];
    LTCBuffer[baseIdx + kIDx_LTC_invM_22] = ltc.invM[2][2];
    LTCBuffer[baseIdx + kIDx_LTC_detM] = ltc.detM;
    LTCBuffer[baseIdx + kIDx_LTC_isIsotropic] = ltc.isIsotropic;

    // Reset Optimization State
    LTCBuffer[baseIdx + kIDx_LTC_Error] = 0.0f;
    LTCBuffer[baseIdx + kIDx_LTC_Iter] = 0.0f;
    LTCBuffer[baseIdx + kIDx_LTC_Lion_m_0] = 0.0f;
    LTCBuffer[baseIdx + kIDx_LTC_Lion_m_1] = 0.0f;
    LTCBuffer[baseIdx + kIDx_LTC_Lion_m_2] = 0.0f;
    LTCBuffer[baseIdx + kIDx_LTC_Adam_v_0] = 0.0f;
    LTCBuffer[baseIdx + kIDx_LTC_Adam_v_1] = 0.0f;
    LTCBuffer[baseIdx + kIDx_LTC_Adam_v_2] = 0.0f;

    // Zero-out remaining buffer entries
    for (int i = kIDx_LTC_Adam_v_2 + 1; i < kLTC_STRIDE; ++i)
    {
        LTCBuffer[baseIdx + i] = 0.0f;
    }
}

/*
Shader Resources:
    Buffer LTCBuffer (as UAV)
*/