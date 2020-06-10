#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>

#include <sys/time.h>
#include <unistd.h>

#include <stdio.h>
#include <algorithm>
#include <fstream>

#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

extern "C" {

static ncnn::Net yolov3;

//struct Object
//{
//  cv::Rect_<float> rect;
//  int label;
//  float prob;
//};

JNIEXPORT jboolean JNICALL
Java_com_example_yolov3tiny_yolov3Tiny_Init(JNIEnv *env, jobject obj, jstring param, jstring bin) {
    __android_log_print(ANDROID_LOG_DEBUG, "yolov3tinyJni", "enter the jni func");

#if NCNN_VULKAN
    yolov3.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    const char *param_path = env->GetStringUTFChars( param, NULL);
    if(param_path == NULL)
        return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "yolov3tinyJni", "load_param %s", param_path);

    int ret = yolov3.load_param(param_path);
    __android_log_print(ANDROID_LOG_DEBUG, "yolov3tinyJni", "load_param result %d", ret);
    env->ReleaseStringUTFChars( param, param_path);

    const char *bin_path = env->GetStringUTFChars( bin, NULL);
    if(bin_path == NULL)
        return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "yolov3tinyJni", "load_model %s", bin_path);

    int ret2 = yolov3.load_model(bin_path);
    __android_log_print(ANDROID_LOG_DEBUG, "yolov3tinyJni", "load_model result %d", ret2);
    env->ReleaseStringUTFChars( bin, bin_path);
    return JNI_TRUE;
}

JNIEXPORT jfloatArray JNICALL Java_com_example_yolov3tiny_yolov3Tiny_Detect(JNIEnv* env, jobject thiz, jobject bitmap)
{
    const int target_size = 416; //input size

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    void* indata;
    AndroidBitmap_lockPixels(env, bitmap, &indata);

//    const char *img_path = env->GetStringUTFChars( imgPath, NULL);
//    if(img_path == NULL)
//        return JNI_FALSE;
//    __android_log_print(ANDROID_LOG_DEBUG, "yolov3tinyJni", "load_img %s", img_path);
//
//    cv::Mat m = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
//    env->ReleaseStringUTFChars( imgPath, img_path);
//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_BGR2RGB, m.cols, m.rows, input->w, input->h);

    //you can resize image in this cpp, or when reading image in java.
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB, width, height, target_size, target_size);
    ncnn::Mat in = ncnn::Mat::from_pixels((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB, width, height);
    __android_log_print(ANDROID_LOG_DEBUG, "yolov3tinyJni", "yolov3_predict_has_input1, in.w: %d; in.h: %d", in.w, in.h);
    AndroidBitmap_unlockPixels(env, bitmap);

    const float norm_vals[3] = {1 / 255.0, 1 / 255.0, 1 / 255.0};
    in.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = yolov3.create_extractor();
    ex.input("data",in);
    ex.set_light_mode(false);
    ex.set_num_threads(4);

    ncnn::Mat out;
    int result = ex.extract("yolo_23", out); //yolo_23 is the out_blob name in param file
    __android_log_print(ANDROID_LOG_DEBUG, "yolov3tinyJni", "extract result %d", result);
   if (result != 0)
        return NULL;

    int output_wsize = out.w;
    int output_hsize = out.h;
    jfloat *output[output_wsize * output_hsize];
    for(int i = 0; i< out.h; i++) {
        for (int j = 0; j < out.w; j++) {
            output[i*output_wsize + j] = &out.row(i)[j];
        }
    }
    jfloatArray jOutputData = env->NewFloatArray(output_wsize * output_hsize);
    if (jOutputData == nullptr) return nullptr;
    env->SetFloatArrayRegion(jOutputData, 0,  output_wsize * output_hsize,
                             reinterpret_cast<const jfloat *>(*output));  // copy

    return jOutputData;

}
}
