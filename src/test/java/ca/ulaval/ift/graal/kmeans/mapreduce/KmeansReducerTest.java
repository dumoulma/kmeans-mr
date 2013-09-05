package ca.ulaval.ift.graal.kmeans.mapreduce;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mrunit.mapreduce.ReduceDriver;
import org.apache.hadoop.mrunit.types.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.runners.MockitoJUnitRunner;

import ca.ulaval.ift.graal.kmeans.mapreduce.KmeansReducer.Counter;

import static org.hamcrest.CoreMatchers.equalTo;

import static org.hamcrest.MatcherAssert.assertThat;

import static org.mockito.Matchers.anyObject;

import static org.mockito.Mockito.verify;

@RunWith(MockitoJUnitRunner.class)
public class KmeansReducerTest {
    private final Map<Integer, Vector> centers = new HashMap<Integer, Vector>();

    private ReduceDriver<IntWritable, VectorWritable, IntWritable, VectorWritable> reduceDriver;

    @Mock
    private Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context context;

    @Before
    public void setUp() throws Exception {
        centers.put(1, new DenseVector(new double[] { 0.0, 0.0 }));
    }

    @Ignore
    public void givenAnIteratorVectorsWillOutputNewCentroid() throws IOException,
            InterruptedException {
        double[] data1 = { 0.1, 0.1 };
        double[] data2 = { -0.2, 0.2 };
        double[] data3 = { 0.0, -0.1 };
        List<VectorWritable> values = new ArrayList<VectorWritable>();
        values.add(new VectorWritable(new DenseVector(data1)));
        values.add(new VectorWritable(new DenseVector(data2)));
        values.add(new VectorWritable(new DenseVector(data3)));

        // when(context.getCounter((Enum<?>) anyObject())).thenCallRealMethod();
        KmeansReducer reducer = new KmeansReducer(centers);
        reducer.reduce(new IntWritable(1), values, context);
        verify(context).write((IntWritable) anyObject(), (VectorWritable) anyObject());
    }

    @Test
    public void givenAnIteratorToCloseVectorsWillOutputConvergedCentroid() throws IOException,
            InterruptedException {
        double[] data1 = { 0.7371, 5.013 };
        double[] data2 = { 0.7372, 5.012 };
        double[] data3 = { 0.7370, 5.014 };
        List<VectorWritable> values = new ArrayList<VectorWritable>();
        values.add(new VectorWritable(new DenseVector(data1)));
        values.add(new VectorWritable(new DenseVector(data2)));
        values.add(new VectorWritable(new DenseVector(data3)));

        KmeansReducer reducer = new KmeansReducer();
        reduceDriver = ReduceDriver.newReduceDriver(reducer);
        Configuration conf = new Configuration();
        conf.set("centroid.path", "data/test/depth_1");
        reduceDriver.withConfiguration(conf);

        reduceDriver.withInputKey(new IntWritable(1));
        reduceDriver.withInputValues(values);
        reduceDriver.getCounters().findCounter(Counter.CONVERGED).setValue(0);

        List<Pair<IntWritable, VectorWritable>> outputs = reduceDriver.run();
        assertThat(outputs.size(), equalTo(1));
        assertThat(reduceDriver.getCounters().findCounter(Counter.CONVERGED).getValue(),
                equalTo(1L));
    }

    @Test
    public void givenAnIteratorToFarVectorsWillOutputNewCentroid() throws IOException,
            InterruptedException {
        double[] data1 = { 3.0, 1.0 };
        double[] data2 = { 2.0, 2.0 };
        double[] data3 = { 1.0, 3.0 };
        List<VectorWritable> values = new ArrayList<VectorWritable>();
        values.add(new VectorWritable(new DenseVector(data1)));
        values.add(new VectorWritable(new DenseVector(data2)));
        values.add(new VectorWritable(new DenseVector(data3)));

        KmeansReducer reducer = new KmeansReducer();
        reduceDriver = ReduceDriver.newReduceDriver(reducer);
        Configuration conf = new Configuration();
        conf.set("centroid.path", "data/test/depth_1");
        reduceDriver.withConfiguration(conf);

        reduceDriver.withInputKey(new IntWritable(1));
        reduceDriver.withInputValues(values);
        reduceDriver.getCounters().findCounter(Counter.CONVERGED).setValue(0);

        Vector expected = new DenseVector(new double[] { 2.0, 2.0 });
        reduceDriver.withOutput(new IntWritable(1), new VectorWritable(expected));
        assertThat(reduceDriver.getCounters().findCounter(Counter.CONVERGED).getValue(),
                equalTo(0L));
    }
}
