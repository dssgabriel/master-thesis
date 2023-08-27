// `a` possede la valeur "Hello, world!"
let a = String::from("Hello, world!");

let a = String::from("Hello, world!");
// La propriete de "Hello, world!" est transferee
// de la variable `a` vers la variable `b`
let b = a;

{
    let a = String::from("Hello, world!");
}
// La valeur "Hello, world!" est automatiquement desalloue
// car son proprietaire `a` est hors de portee

