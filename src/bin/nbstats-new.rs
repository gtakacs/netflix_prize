use std::env;
use netflix_prize::{nbstats, SPLIT_NEW};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    let (rtg, stat) = job_name.rsplit_once('_')
        .unwrap_or_else(|| panic!("job name must be of form <rtg>_<stat>: {}", job_name));

    nbstats::save_nbstats_split(rtg, stat, SPLIT_NEW);
}
