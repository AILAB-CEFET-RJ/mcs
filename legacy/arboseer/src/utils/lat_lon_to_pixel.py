from osgeo import osr, ogr

def latlon_to_pixel(lat, lon, dataset):
    # Get the GeoTransform vector
    geo_transform = dataset.GetGeoTransform()
    source_sr = osr.SpatialReference()
    source_sr.ImportFromWkt(dataset.GetProjection())

    target_sr = osr.SpatialReference()
    target_sr.ImportFromEPSG(4326)  # WGS84

    transform = osr.CoordinateTransformation(target_sr, source_sr)

    # Transform the point coordinates
    point = ogr.CreateGeometryFromWkt(f"POINT ({lon} {lat})")
    point.Transform(transform)

    # Convert geographic coordinates to pixel coordinates
    x = (point.GetX() - geo_transform[0]) / geo_transform[1]
    y = (point.GetY() - geo_transform[3]) / geo_transform[5]

    return int(x), int(y)