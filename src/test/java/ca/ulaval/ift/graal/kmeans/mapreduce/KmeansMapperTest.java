package ca.ulaval.ift.graal.kmeans.mapreduce;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.runners.MockitoJUnitRunner;

import static org.mockito.Mockito.verify;

@RunWith(MockitoJUnitRunner.class)
public class KmeansMapperTest {
    private final Map<Integer, Vector> centers = new HashMap<Integer, Vector>();

    @Mock
    private Mapper<LongWritable, VectorWritable, IntWritable, VectorWritable>.Context context;

    // private MapDriver<LongWritable, VectorWritable, IntWritable,
    // VectorWritable> mapDriver;
    //
    @Before
    public void setUp() throws Exception {
        // mapDriver = MapDriver.newMapDriver(new KmeansMapper());
        double[] center1 = { 0.0, 0.0 };
        double[] center2 = { 2.0, 2.0 };
        double[] center3 = { 3.0, 3.0 };
        centers.put(1, new DenseVector(center1));
        centers.put(2, new DenseVector(center2));
        centers.put(3, new DenseVector(center3));
    }

    @Test
    public void givenAVectorWillOutputTheNearestCluster() throws IOException, InterruptedException {
        LongWritable key = new LongWritable(1);
        double[] data = { 0.5, 0.5 };
        VectorWritable value = new VectorWritable(new DenseVector(data));

        KmeansMapper mapper = new KmeansMapper(centers);
        mapper.map(key, value, context);
        verify(context).write(new IntWritable(1), value);
    }
}
