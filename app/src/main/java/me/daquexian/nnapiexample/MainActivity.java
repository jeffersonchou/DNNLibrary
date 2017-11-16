package me.daquexian.nnapiexample;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;

import me.daquexian.dnnlibrary.ModelWrapper;

public class MainActivity extends AppCompatActivity {

    @SuppressWarnings("unused")
    private static final String TAG = "NNAPI Example";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        float[] result = ModelWrapper.init(getAssets());
    }
}
