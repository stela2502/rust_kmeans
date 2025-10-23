//main.rs
use clap::Parser;
use my_kmeans_lib::DataSet;

#[derive(Parser)]
#[clap(version = "0.2.0", author = "Stefan L. <stefan.lang@med.lu.se>")]
struct Opts {
    #[clap(short, long)]
    file: String,

    #[clap(short, long)]
    k: usize,

    #[clap(short, long)]
    outfile: String,
}

fn main() -> anyhow::Result<()> {
    let opts = Opts::parse();

    let ds = DataSet::from_tsv(&opts.file)?;
    println!(
        "Loaded {} rows × {} columns",
        ds.data.nrows(),
        ds.data.ncols()
    );

    let clusters = ds.kmeans3d(opts.k, 50)?;
    println!("Assigned {} points into {} clusters", clusters.len(), opts.k);

    std::fs::write(&opts.outfile, clusters.iter().map(|c| c.to_string()).collect::<Vec<_>>().join("\n"))?;

    Ok(())
}