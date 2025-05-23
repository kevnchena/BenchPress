from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(storage={'root_dir': './bench_press_images'})

google_crawler.crawl(
    keyword='bench press',
    max_num=500,
    min_size=(300, 300),
    file_idx_offset=0
)