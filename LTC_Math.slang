#ifndef LTC_MATH_SLANG
#define LTC_MATH_SLANG

static const float c_pi = 3.14159265359f;

struct LTCParams : IDifferentiable
{
    float m11;
    float m22;
    float m13;
};

struct LTC : IDifferentiable
{
    // lobe magnitude
    float magnitude;

    // Average Schlick Fresnel term
    float fresnel;

    // parametric representation
    float m11, m22, m13;
    float3 X, Y, Z;

    // matrix representation
    float3x3 M;
    float3x3 invM;
    float detM;

    // 1.0f or 0.0f (isotropic or anisotropic)
    no_diff float isIsotropic;
};

[Differentiable]
void LTC_update(inout LTC ltc) // compute matrix from parameters
{
    // M = R * S
    // R has columns X, Y, Z.
    // HLSL float3x3(X, Y, Z) creates a matrix with ROWS X, Y, Z.
    // So float3x3(X, Y, Z) is R^T.
    float3x3 R_t = float3x3(ltc.X, ltc.Y, ltc.Z);
    float3x3 R = transpose(R_t);

    // S = [m11 0 m13; 0 m22 0; 0 0 1] (Column-major construction in C++ GLM)
    // In HLSL float3x3 constructor is row-major.
    float3x3 S = float3x3(
        ltc.m11, 0, ltc.m13,
        0, ltc.m22, 0,
        0, 0, 1
    );

    ltc.M = mul(R, S);
    
    ltc.detM = abs(determinant(ltc.M));

    // invM = S^-1 * R^T
    // S^-1
    // Clamp to prevent division by near-zero values
    float safe_m11 = max(ltc.m11, 0.0001f);
    float safe_m22 = max(ltc.m22, 0.0001f);
    float3x3 invS = float3x3(
        1.0f / safe_m11, 0, -ltc.m13 / safe_m11,
        0, 1.0f / safe_m22, 0,
        0, 0, 1);

    // R^T is R_t
    ltc.invM = mul(invS, R_t);
}

[Differentiable]
float LTC_eval(in LTC ltc, no_diff float3 L)
{
    float3 Loriginal = normalize(mul(ltc.invM, L));
    float3 L_ = mul(ltc.M, Loriginal);

    // Clamp to prevent division by near-zero values
    float l = max(length(L_), 1e-5f);
    float safe_detM = max(ltc.detM, 1e-5f);
    float Jacobian = safe_detM / (l*l*l);

    float D = 1.0f / c_pi * max(0.0f, Loriginal.z);

    float res = ltc.magnitude * D / Jacobian;
    // Clamp output to prevent extreme values
    return min(res, 1e6f);
}

[Differentiable]
float3 LTC_sample(in LTC ltc, no_diff float U1, no_diff float U2)
{
    const float theta = acos(sqrt(U1));
    const float phi = 2.0f*c_pi * U2;
    const float3 L = normalize(mul(ltc.M, float3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))));
    return L;
}

// GGX BRDF
// V: View vector (pointing away from surface)
// L: Light vector (pointing away from surface)
// alpha: roughness^2
[Differentiable]
float GGX_eval(no_diff float3 V, no_diff float3 L, no_diff float alpha, no_diff inout float outPdf)
{
    if (V.z <= 0.0f || L.z <= 0.0f) return 0.0f;

    float3 H = normalize(V + L);
    float NdotH = H.z;
    float NdotV = V.z;

    // D (GGX/Trowbridge-Reitz)
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH * alpha2 + (1.0f - NdotH * NdotH));
    float D = alpha2 / (c_pi * denom * denom);

    // G (Smith)
    float NdotV2 = NdotV * NdotV;
    float tan2V = max(0.0f, 1.0f - NdotV2) / NdotV2;
    float lambdaV = 0.5f * (-1.0f + sqrt(1.0f + alpha2 * tan2V));

    float NdotL = L.z;
    float NdotL2 = NdotL * NdotL;
    float tan2L = max(0.0f, 1.0f - NdotL2) / NdotL2;
    float lambdaL = 0.5f * (-1.0f + sqrt(1.0f + alpha2 * tan2L));

    float G2 = 1.0f / (1.0f + lambdaV + lambdaL);

    outPdf = abs(D * NdotH / 4.0f / dot(V, H));
    return D * G2 / (4.0f * NdotV);
}

[Differentiable]
float3 GGX_sample(no_diff float3 V, no_diff float alpha, no_diff float U1, no_diff float U2)
{
    const float phi = 2.0f*3.14159f * U1;
    const float r = alpha*sqrt(U2/(1.0f - U2));
    const float3 N = normalize(float3(r*cos(phi), r*sin(phi), 1.0f));
    const float3 L = -V + 2.0f * N * dot(N, V);
    return L;
}

#endif