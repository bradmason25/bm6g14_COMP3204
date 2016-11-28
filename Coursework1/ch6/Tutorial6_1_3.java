package uk.ac.soton.ecs.bm6g14.ch6;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.dataset.BingImageDataset;
import org.openimaj.util.api.auth.DefaultTokenFactory;
import org.openimaj.util.api.auth.common.BingAPIToken;

public class Tutorial6_1_3 {
	public static void main(String[] args) {
		try {
			
			BingAPIToken bingtoken = DefaultTokenFactory.get(BingAPIToken.class);
			BingImageDataset<FImage> books = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "book", 10);
			DisplayUtilities.display("Books", books);
			
		} catch(Exception e) { e.printStackTrace(); }
	}
}
