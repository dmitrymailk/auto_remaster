#include "ReShade.fxh"

texture Screen1
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
};
sampler Screen1_sampler { Texture = Screen1; };

texture Screen2
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
};
sampler Screen2_sampler { Texture = Screen2; };

texture Screen3
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
};
sampler Screen3_sampler { Texture = Screen3; };

#include "ReShade.fxh"

uniform int iUIBackground <
    ui_type = "combo";
ui_label = "Background Type";
ui_items = "Game Image\0Solid Color\0";
> = 1;

uniform float3 fUIColorBackground <
    ui_type = "color";
ui_label = "Background Color";
> = float3(0.0, 0.0, 0.0);

uniform float3 fUIColorLines <
    ui_type = "color";
ui_label = "Wireframe Color";
> = float3(1.0, 1.0, 1.0);

uniform float fUIStrength <
    ui_type = "drag";
ui_label = "Edge Strength";
ui_min = 0.0;
ui_max = 10.0;
ui_step = 0.1;
> = 1.0;

uniform float fUIThreshold <
    ui_type = "drag";
ui_label = "Edge Threshold";
ui_min = 0.0;
ui_max = 1.0;
ui_step = 0.001;
> = 0.00;

uniform float fUIDepthMultiplier <
    ui_type = "drag";
ui_label = "Depth Multiplier";
ui_min = 1.0;
ui_max = 100.0;
ui_step = 0.1;
> = 10.0;

uniform bool bUIUseDepthEdges <
    ui_label = "Use Depth Edges";
> = true;

uniform bool bUIUseColorEdges <
    ui_label = "Use Color Edges";
> = true;

uniform float fUIDepthThreshold <
    ui_type = "drag";
ui_label = "Depth Edge Threshold";
ui_min = 0.0;
ui_max = 1.0;
ui_step = 0.001;
> = 0.1;

// Sobel edge detection
float2 SobelSample(float2 texcoord, float2 offset)
{
    float4 center = tex2D(ReShade::BackBuffer, texcoord);
    float4 left = tex2D(ReShade::BackBuffer, texcoord - float2(offset.x, 0));
    float4 right = tex2D(ReShade::BackBuffer, texcoord + float2(offset.x, 0));
    float4 up = tex2D(ReShade::BackBuffer, texcoord - float2(0, offset.y));
    float4 down = tex2D(ReShade::BackBuffer, texcoord + float2(0, offset.y));

    float2 sobel;
    sobel.x = length((right - left).rgb);
    sobel.y = length((up - down).rgb);
    return sobel;
}

float3 MeshEdges_PS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 original = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float edge = 0;

    if (bUIUseColorEdges)
    {
        float2 sobelEdge = SobelSample(texcoord, ReShade::PixelSize);
        edge = length(sobelEdge);
    }

    if (bUIUseDepthEdges)
    {
        float depth = ReShade::GetLinearizedDepth(texcoord) * fUIDepthMultiplier;
        float depthEdge = 0;

        float2 offsets[4] = {
            float2(1, 0),
            float2(-1, 0),
            float2(0, 1),
            float2(0, -1)};

        for (int i = 0; i < 4; i++)
        {
            float neighborDepth = ReShade::GetLinearizedDepth(texcoord + offsets[i] * ReShade::PixelSize) * fUIDepthMultiplier;
            depthEdge = max(depthEdge, abs(depth - neighborDepth));
        }

        edge = max(edge, depthEdge > fUIDepthThreshold ? depthEdge : 0);
    }

    edge = edge > fUIThreshold ? edge * fUIStrength : 0;
    edge = saturate(edge);

    float3 background = iUIBackground == 0 ? original : fUIColorBackground;
    return lerp(background, fUIColorLines, edge);
}

float4 Pass_Screen(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return tex2D(ReShade::BackBuffer, texcoord);
}

float4 Pass_All(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    if (texcoord.x < 0.5)
    {
        if (texcoord.y < 0.5)
        {
            return tex2D(Screen1_sampler, float2(texcoord.x * 2.0, texcoord.y * 2.0));
        }
        else
        {
            return tex2D(Screen3_sampler, float2(texcoord.x * 2.0, (texcoord.y - 0.5) * 2.0));
        }
    }
    else
    {
        if (texcoord.y < 0.5)
        {
            return tex2D(Screen2_sampler, float2((texcoord.x - 0.5) * 2.0, texcoord.y * 2.0));
        }
        else
        {
            return tex2D(ReShade::BackBuffer, float2((texcoord.x - 0.5) * 2.0, (texcoord.y - 0.5) * 2.0));
        }
    }
}



technique Screen_1
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Pass_Screen;
        RenderTarget = Screen1;
    }
}

technique Screen_2
{
    pass
    {
        VertexShader = PostProcessVS;
        // PixelShader = Pass_Screen;
        PixelShader = MeshEdges_PS;
        RenderTarget = Screen2;
    }
}

technique Screen_3
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Pass_Screen;
        // PixelShader = PS_Depth;
        RenderTarget = Screen3;
    }
}

technique Screen_4
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Pass_All;
    }
}
