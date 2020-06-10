package com.example.yolov3tiny;
import android.graphics.Bitmap;

public class yolov3Tiny {
    public native boolean Init(String param, String bin);
    public native float[] Detect(Bitmap bitmap);
    static {
        System.loadLibrary("yolov3_tiny_jni");
    }

}
