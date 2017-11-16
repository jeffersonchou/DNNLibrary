package me.daquexian.dnnlibrary;

import android.content.res.AssetManager;

/**
 * Created by daquexian on 2017/11/12.
 * Java wrapper
 */

public class ModelWrapper {

    static {
        System.loadLibrary( "dnnlibrary");
    }

    public static native float[] init(AssetManager assetManager);
}
