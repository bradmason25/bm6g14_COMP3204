package Run2;

import java.io.IOException;
import java.io.InputStream;

import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.io.InputStreamObjectReader;

public class IdentifiableFImageReader implements InputStreamObjectReader<IdentifiableObject<FImage>> {
	
	@Override
	public IdentifiableObject<FImage> read( InputStream source ) throws IOException {
		FImage img = ImageUtilities.FIMAGE_READER.read( source );
		return new IdentifiableObject<FImage>( String.valueOf( img.hashCode() ), img );
	}
	
	@Override
	public boolean canRead( InputStream stream, String name ) {
		return ImageUtilities.FIMAGE_READER.canRead( stream, name );
	}
	
}