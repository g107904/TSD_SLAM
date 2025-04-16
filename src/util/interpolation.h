#pragma once

namespace lsd_slam
{
    template <typename T>
    inline T get_interpolated_element(const T* const source, const float x, const float y, const int width)
    {
        int ix = (int)x;
        int iy = (int)y;
        float dx = x - ix;
        float dy = y - iy;
	float dxdy = dx*dy;

        T result = (dxdy) * (*(source+ix+1+(iy+1)*width))
                + (dy - dxdy) * (*(source+ix+(iy+1)*width))
                + (dx - dxdy) * (*(source+ix+1+iy*width))
                + (1 - dx - dy + dxdy) * (*(source + ix + iy * width));
        return result;
    }
}
