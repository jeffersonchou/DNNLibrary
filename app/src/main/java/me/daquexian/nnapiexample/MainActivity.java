package me.daquexian.nnapiexample;

import android.Manifest;
import android.content.res.AssetManager;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.List;

import pub.devrel.easypermissions.EasyPermissions;

public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener2,
        EasyPermissions.PermissionCallbacks {

    @SuppressWarnings("unused")
    private static final String TAG = "NNAPI Example";
    private static final int INPUT_LENGTH = 28;

    String[] params = {Manifest.permission.CAMERA};

    private TextView textView;
    private EnhancedCameraView cameraView;

    static {
        OpenCVLoader.initDebug();
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = findViewById(R.id.text);
        cameraView = findViewById(R.id.camera_view);

        textView.setTextSize(20);
        textView.setText(R.string.welcome_message);

        if (EasyPermissions.hasPermissions(this, params)) {
            initModel(getAssets());

            initCamera();
        } else {
            EasyPermissions.requestPermissions(this, "Please grant or the app can't init",
                    321, params);
        }
    }

    private void initCamera() {
        cameraView.setCvCameraViewListener(this);
        cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
        cameraView.enableView();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat grayscale = inputFrame.gray();
        float[] inputData = getInputDataFromGrayscaleImageMat(grayscale);
        final int predictNumber = predict(inputData);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                textView.setText(getResources().getString(R.string.predict_text, predictNumber));
            }
        });
        return grayscale;
    }

    private float[] getInputDataFromGrayscaleImageMat(Mat imageMat) {
        Mat inputDataMat = new Mat();

        // convert the image to 28 * 28, grayscale, 0~1, and smaller means whiter
        imageMat = centerCropAndScale(imageMat, INPUT_LENGTH);
        imageMat.convertTo(imageMat, CvType.CV_32F, 1. / 255);
        Core.subtract(Mat.ones(imageMat.size(), CvType.CV_32F), imageMat, inputDataMat);

        float[] inputData = new float[inputDataMat.width() * inputDataMat.height()];

        inputDataMat.get(0, 0, inputData);

        return inputData;
    }

    private Mat centerCropAndScale(Mat mat, int length) {
        Mat _mat = mat.clone();
        if (_mat.height() > _mat.width()) {
            _mat = new Mat(_mat, new Rect(0, (_mat.height() - _mat.width()) / 2, _mat.width(), _mat.width()));
            Imgproc.resize(_mat, _mat, new Size(length, length));
        } else {
            _mat = new Mat(_mat, new Rect((_mat.width() - _mat.height()) / 2, 0, _mat.height(), _mat.height()));
            Imgproc.resize(_mat, _mat, new Size(length, length));
        }
        return _mat;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        clearModel();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        // Forward results to EasyPermissions
        EasyPermissions.onRequestPermissionsResult(requestCode, permissions, grantResults, this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public void onPermissionsGranted(int requestCode, List<String> perms) {
        recreate();
    }

    @Override
    public void onPermissionsDenied(int requestCode, List<String> perms) {
        finish();
    }

    public native void initModel(AssetManager assetManager);
    public native int predict(float[] data);
    public native int clearModel();
}
