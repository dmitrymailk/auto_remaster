// i think this is better but it doesn't work in nfs mw 2005 by default
// https://reshade.me/forum/shader-presentation/4635-mesh-edges

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

technique MeshEdges
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = MeshEdges_PS;
    }
}
