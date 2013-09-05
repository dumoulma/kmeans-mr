package ca.ulaval.ift.graal.kmeans.mapreduce;

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;

import static org.hamcrest.MatcherAssert.assertThat;

public class MRUtilsTest {
    private final Configuration conf = new Configuration();

    @Test
    public void givenSeqFileShouldReturnCorrectNumberOfClusters() throws IOException {
        conf.set("centroid.path", "data/test/centroid.seq");
        Map<Integer, Vector> centers = MRUtils.readClusters(conf);
        assertThat(centers.size(), is(4));
    }

    @Test
    public void givenDirShouldReturnCorrectNumberOfClusters() throws IOException {
        conf.set("centroid.path", "data/test/depth_1");
        Map<Integer, Vector> centers = MRUtils.readClusters(conf);
        assertThat(centers.size(), is(4));
    }
}
