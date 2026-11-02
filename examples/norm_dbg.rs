fn main() {
    for t in [
        "On March 4th, 2026, NASA paid $1,250.75 for 3.5 hours of work.",
        "CPU vs GPU cost $5.",
        "He was born on 2001-11-03 and drank 1.5 liters at 5%, plus tax & fees for $0.99.",
        "NASA and CPU",
        "Hello, world!",
    ] {
        match pivo_runner::text::normalize(t) {
            Ok(n) => println!("{:?}\n  -> {:?}\n", t, n),
            Err(e) => println!("{:?}\n  -> ERR {}\n", t, e),
        }
    }
}
