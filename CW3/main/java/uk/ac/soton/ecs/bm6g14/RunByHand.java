package uk.ac.soton.ecs.bm6g14;

import java.io.FileOutputStream;
import java.io.PrintWriter;

import org.apache.commons.vfs2.FileObject;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

public class RunByHand {
	public static void main(String[] args) {
		PrintWriter out = null;
		try {
			Run1 run1 = new Run1();			
			
			VFSListDataset<FImage> testData = new VFSListDataset<FImage>("zip:file:/home/brad/OpenIMAJ_Coursework3/testing.zip", ImageUtilities.FIMAGE_READER);
			
			int k = 4;
			FileObject[] files = testData.getFileObjects();
			
			out = new PrintWriter("run1.txt");
			
			for(int i=0;i<testData.size();i++) {
				String clas = run1.knn(k, testData.get(i));
				String filename = files[i].getName().getBaseName();
				out.println(filename+" "+clas);
			}
			out.close();
		} catch (Exception e) {
			out.close();
			e.printStackTrace();
		}
	}
}