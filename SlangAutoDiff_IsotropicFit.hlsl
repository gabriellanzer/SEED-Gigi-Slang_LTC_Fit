// SlangAutoDiff technique, shader Descend
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

// Optimization State Layout
static const int kIDx_LTC_Error = 34;
static const int kIDx_LTC_Iter = 35;
static const int kIDx_LTC_Lion_m_0 = 36;
static const int kIDx_LTC_Lion_m_1 = 37;
static const int kIDx_LTC_Lion_m_2 = 38;

// Random number generator
uint hash(uint x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

[Differentiable]
float3 SampleCosine(no_diff float u1, no_diff float u2, no_diff out float pdf)
{
    float r = sqrt(u1);
    float theta = 2.0 * c_pi * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u1));
    pdf = z / c_pi;
    return float3(x, y, z);
}

[Differentiable]
float ComputeLoss(LTCParams params, no_diff float3 V, no_diff float3 X, no_diff float3 Y, no_diff float3 Z, no_diff float alpha, no_diff float magnitude, no_diff int sampleCount)
{
    LTC ltc;
    ltc.magnitude = magnitude;
    ltc.fresnel = 0.0f;
    ltc.X = X;
    ltc.Y = Y;
    ltc.Z = Z;
    ltc.m11 = params.m11;
    ltc.m22 = params.m22;
    ltc.m13 = params.m13;
    
    // Initialize matrices to zero (will be overwritten by update)
    ltc.M = float3x3(0,0,0, 0,0,0, 0,0,0);
    ltc.invM = float3x3(0,0,0, 0,0,0, 0,0,0);
    ltc.detM = 0.0f;
    ltc.isIsotropic = 0.0f;

	LTC_update(ltc);

	float error = 0.0f;

	// Multiple Importance Sampling: sample both LTC and BRDF
	[MaxIters(128)]
	for (int j = 0; j < sampleCount; ++j)
	{
		[MaxIters(128)]
		for (int i = 0; i < sampleCount; ++i)
		{
			const float u1 = (i + 0.5f) / sampleCount;
			const float u2 = (j + 0.5f) / sampleCount;

			// Sample 1: Importance sample LTC
			{
				float3 L = LTC_sample(ltc, u1, u2);

				float pdf_brdf;
				float eval_brdf = GGX_eval(V, L, alpha, pdf_brdf);
				float eval_ltc = LTC_eval(ltc, L);
				float pdf_ltc = eval_ltc / ltc.magnitude;

				// MIS weight with balance heuristic
				float error_ = abs(eval_brdf - eval_ltc);
				error_ = error_ * error_ * error_;
            	error += error_/(pdf_ltc + pdf_brdf);
			}

			// Sample 2: Importance sample BRDF
			{
				float3 L = GGX_sample(V, alpha, u1, u2);

				float pdf_brdf;
				float eval_brdf = GGX_eval(V, L, alpha, pdf_brdf);
				float eval_ltc = LTC_eval(ltc, L);
				float pdf_ltc = eval_ltc / ltc.magnitude;

				// MIS weight with balance heuristic
				float error_ = abs(eval_brdf - eval_ltc);
				error_ = error_ * error_ * error_;
            	error += error_/(pdf_ltc + pdf_brdf);
			}
		}
	}

	return error / float(sampleCount * sampleCount);
}

[shader("compute")]
/*$(_compute:csmain)*/(uint3 DTid : SV_DispatchThreadID)
{
    int t = DTid.x;
    int a = DTid.y;
    int2 tableSize = /*$(Variable:TableSize)*/;

	if (t >= tableSize.x || a >= tableSize.y)
		return;

	int index = a * tableSize.x + t;
	int baseIdx = index * kLTC_STRIDE;

	// Read Iteration
	float iter = LTCBuffer[baseIdx + kIDx_LTC_Iter];

	// Target
	float roughness = float(a) / float(tableSize.y - 1);
	float alpha = max(roughness * roughness, 0.00001f);

	float x = float(t) / float(tableSize.x - 1);
	float ct = 1.0f - x * x;
	float theta = min(c_pi/2.0f, acos(ct));

	float sinT = sin(theta);
	float cosT = max(cos(theta), 1e-7f);
	float3 V = float3(sinT, 0.0f, cosT);

	// Load Params
	LTCParams params;
	params.m11 = LTCBuffer[baseIdx + kIDx_LTC_m11];
	params.m22 = LTCBuffer[baseIdx + kIDx_LTC_m22];
	params.m13 = LTCBuffer[baseIdx + kIDx_LTC_m13];

	float magnitude = LTCBuffer[baseIdx + kIDx_LTC_Magnitude];
	float3 X = float3(LTCBuffer[baseIdx + kIDx_LTC_X_0], LTCBuffer[baseIdx + kIDx_LTC_X_1], LTCBuffer[baseIdx + kIDx_LTC_X_2]);
	float3 Y = float3(LTCBuffer[baseIdx + kIDx_LTC_Y_0], LTCBuffer[baseIdx + kIDx_LTC_Y_1], LTCBuffer[baseIdx + kIDx_LTC_Y_2]);
	float3 Z = float3(LTCBuffer[baseIdx + kIDx_LTC_Z_0], LTCBuffer[baseIdx + kIDx_LTC_Z_1], LTCBuffer[baseIdx + kIDx_LTC_Z_2]);

	// AutoDiff
	var dp_params = diffPair(params);

    int sampleCount = /*$(Variable:SampleCount)*/;
	bwd_diff(ComputeLoss)(dp_params, V, X, Y, Z, alpha, magnitude, sampleCount, 1.0f);

	// Lion Update
	{
		float lr = pow(/*$(Variable:LearningRate)*/, /*$(Variable:LearningRatePower)*/);
		float weightDecay = /*$(Variable:WeightDecay)*/;
		float beta1 = 0.9f;
		float beta2 = 0.99f;

		// Clip gradients to prevent extreme values
		float grads[3] = {
			clamp(dp_params.d.m11, -10.0f, 10.0f),
			clamp(dp_params.d.m22, -10.0f, 10.0f),
			clamp(dp_params.d.m13, -10.0f, 10.0f)
		};

		// Read Lion state (reusing Adam m slots)
		float m[3];
		m[0] = LTCBuffer[baseIdx + kIDx_LTC_Lion_m_0];
		m[1] = LTCBuffer[baseIdx + kIDx_LTC_Lion_m_1];
		m[2] = LTCBuffer[baseIdx + kIDx_LTC_Lion_m_2];

		float p[3] = {params.m11, params.m22, params.m13};

		for (int i = 0; i < 3; ++i)
		{
			float g = grads[i];

			// Lion update
			float c = beta1 * m[i] + (1.0f - beta1) * g;
			p[i] -= lr * (sign(c) + weightDecay * p[i]);
			m[i] = beta2 * m[i] + (1.0f - beta2) * g;

			// Increased threshold to prevent near-singular matrices
			if (i == 0 || i == 1)
				p[i] = max(p[i], 0.0001f);
		}

		// Enforce isotropy for t=0
		if (t == 0)
		{
			p[2] = 0.0f; // m13
			p[1] = p[0]; // m22 = m11
		}

		// Write back Lion state
		LTCBuffer[baseIdx + kIDx_LTC_Lion_m_0] = m[0];
		LTCBuffer[baseIdx + kIDx_LTC_Lion_m_1] = m[1];
		LTCBuffer[baseIdx + kIDx_LTC_Lion_m_2] = m[2];
		
		// Update params
		params.m11 = p[0];
		params.m22 = p[1];
		params.m13 = p[2];
	}

	// Write back params
	LTCBuffer[baseIdx + kIDx_LTC_m11] = params.m11;
	LTCBuffer[baseIdx + kIDx_LTC_m22] = params.m22;
	LTCBuffer[baseIdx + kIDx_LTC_m13] = params.m13;

	// Compute Error
	float error = ComputeLoss(params, V, X, Y, Z, alpha, magnitude, sampleCount);
	// Guard against NaN/Inf values
	if (isnan(error) || isinf(error))
		error = 1e6f;
	LTCBuffer[baseIdx + kIDx_LTC_Error] = error;
	LTCBuffer[baseIdx + kIDx_LTC_Iter] = iter + 1.0f;

	// Update LTC matrices and write back
	LTC ltc;
	ltc.magnitude = magnitude;
	ltc.fresnel = LTCBuffer[baseIdx + kIDx_LTC_Fresnel];
	ltc.X = X;
	ltc.Y = Y;
	ltc.Z = Z;
	ltc.m11 = params.m11;
	ltc.m22 = params.m22;
	ltc.m13 = params.m13;

	// Initialize matrices to zero
	ltc.M = float3x3(0,0,0, 0,0,0, 0,0,0);
    ltc.invM = float3x3(0,0,0, 0,0,0, 0,0,0);
	ltc.detM = 0.0f;
	ltc.isIsotropic = 0.0f;

	LTC_update(ltc);

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
}

/*
Shader Resources:
    Buffer LTCBuffer (as UAV)
*/
