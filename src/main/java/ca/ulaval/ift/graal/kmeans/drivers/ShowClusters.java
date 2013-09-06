package ca.ulaval.ift.graal.kmeans.drivers;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import ca.ulaval.ift.graal.utils.Util;

public class ShowClusters {
    public static void main(String[] args) throws IOException {
        Configuration conf = new Configuration();
        Util.showClusters(conf, new Path("data/kmeans/depth_1"));
    }
}
