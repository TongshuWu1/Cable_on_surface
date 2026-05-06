try:
    from .trackdlo_spatial_pipeline import main
except ImportError:
    from trackdlo_spatial_pipeline import main


if __name__ == "__main__":
    main()
