var express = require('express');
var session = require('express-session');
var MySQLStore = require('express-mysql-session')(session);
var bodyParser = require('body-parser');
var bkfd2Password = require("pbkdf2-password");
var passport = require('passport');
var LocalStrategy = require('passport-local').Strategy;
var hasher = bkfd2Password();
var mysql = require('mysql');
var multer = require('multer');
var shellParser = require('node-shell-parser');
var fs = require('fs')

var storage = multer.diskStorage({
    destination: function(req, file, cb){
        cb(null, 'public/uploads/')
    },
    filename: function(req, file, cb){
        cb(null, (file.originalname.split('.')[0])+'_'+Date.now()+'.jpg') 
    }
})
var upload = multer({ storage: storage })
var loginfail = 0;

var conn = mysql.createConnection({
  host     : 'localhost',
  user     : 'root',
  password : 'vmfkdlsgo1',
  database : 'log',
  multipleStatements: true
});
conn.connect();

var app = express();
app.set('views', './page')
app.set('view engine', 'pug')
app.use(express.static('public'))
app.use(express.static('public/mask_output'))
app.use(express.static('public/gan_output'))
app.use(express.static('public/stylesheets'))
app.use(express.static('public/uploads'))
// app.use('/upload', express.static('uploads'))
// app.use('/process', express.static('output'))
app.use(bodyParser.urlencoded({ extended: false }));
app.use(session({
  secret: '1234DSFs@adf1234!@#$asd',
  resave: false,
  saveUninitialized: true,
  store:new MySQLStore({
    host:'localhost',
    port:3306,
    user:'root',
    password:'vmfkdlsgo1',
    database:'log'
  })
}));
app.use(passport.initialize());
app.use(passport.session());

app.listen(3003, function(){
    console.log('Connected 3003 port!!!');
});

app.get('/welcome', function(req, res){
  if(req.user && req.user.displayName) {
    // here is main page
    //res.render('show', {username: req.user.username, displayName: req.user.displayName});
    res.redirect('/upload')
  } 
  // 아이디 비밀번호 잘못 입력
  else {
    loginfail = 1;
    res.redirect('/auth/login')
  }
});

app.get('/upload', function(req, res){
    var sql = 'select id, image from image where username=?'
    conn.query(sql, [req.user.username], (err, rows)=>{
        if(err) console.error(err);
        else{
            console.log(rows)
            res.render('show', {file: rows})
        }
    })
})
//upload.single('avatar')-> 사용자에게 받아온 데이터중 파일이 있으면 req object 안에서 file property 사용 가능
app.post('/upload', upload.single('userfile'), function(req, res){
    console.log(req.file);
    //res.send('uploaded: ' + req.file.filename)
    console.log(req.user.username);
    var sql = 'insert into image(username, image) values(?, ?);'+'select id, image from image where username=?'
    conn.query(sql, [req.user.username, req.file.filename, req.user.username], (err, rows)=>{
        if(err) console.log(err);
        console.log('rows[1]:', rows[1], rows[1].length)
        res.render('show', {file: rows[1]})
    })
})



passport.serializeUser(function(user, done) {
  console.log('serializeUser', user);
  done(null, user.authId);
});
passport.deserializeUser(function(id, done) {
  console.log('deserializeUser', id);
  var sql = 'select * from users where authId=?';
  conn.query(sql, [id], function(err, results){
      if(err){
          console.log(err)
          //done('There is no user');
          return done(null, false);
      }else{
          done(null, results[0]);
      }
  })
});
passport.use(new LocalStrategy(
  function(username, password, done){
    var uname = username;
    var pwd = password;
    var sql = 'select * from users where authId = ?';
    conn.query(sql, ['local:'+uname], function(err, results){
        console.log(results);
        if(!results[0]){
            //return done('there is no user')
            return done(null, false);
        }
        var user = results[0]
        return hasher({password:pwd, salt:user.salt}, function(err, pass, salt, hash){
            if(hash === user.password){
                console.log('LocalStrategy', user)
                done(null, user);
            }else{
                done(null, false);
            }
        })
    });
  }
));

app.get('/auth/login', function(req, res){
    if(loginfail === 1){
        res.render('login', {loginfail: 1});
        loginfail = 0;
    }
    else{
        res.render('login', {loginfail: 0});
    }
});
app.post(
  '/auth/login',
  passport.authenticate(
    'local',
    {
      successRedirect: '/welcome',
      failureRedirect: '/welcome',
      failureFlash: false
    }
  )
);
app.get('/auth/logout', function(req, res){
    req.logout();
    req.session.save(function(){
      loginfail = 0;
      res.redirect('/auth/login');
    });
});

var re_register = {username: "", displayName: "", email: "", password: ""}
app.get('/auth/register', function(req, res){
    console.log(re_register)
    res.render('register', {info: re_register});
});
app.post('/auth/register', function(req, res){
    if(req.body.password != req.body.re_password){
        re_register = {
            username:req.body.username,
            displayName:req.body.displayName,
            email:req.body.email,
            password:req.body.password
        }
        res.redirect('/auth/register')
    }
    else{
        re_register = {username: "", displayName: "", email: "", password: ""}
        hasher({password:req.body.password}, function(err, pass, salt, hash){
            var user = {
                authId:'local:'+req.body.username,
                username:req.body.username,
                password:hash,
                salt:salt,
                displayName:req.body.displayName,
                email:req.body.email
            };
            var sql = 'INSERT INTO users SET ?';
            conn.query(sql, user, function(err, results){
                if(err){
                    console.log(err);
                    res.status(500);
                } else {
                    req.login(user, function(err){
                        req.session.save(function(){
                            res.redirect('/welcome');
                        })
                    })
                }
            });
        });
    }
});

app.get('/process/:id', (req, res)=>{
    var id = req.params.id;
    var sql = "select id, image from image where id=? and username=?";
    conn.query(sql, [id, req.user.username], (err, rows)=>{
        if(err) console.error(err);
        else{           
            console.log('original image name: ' + rows[0].image);
            original_image = rows[0].image
            var shellOutput = '';
            const spawn = require('child_process').spawn
            const result = spawn('python', ['.\\Mask_RCNN\\samples\\balloon\\balloon.py', 'splash', '--weights=.\\Mask_RCNN\\logs\\mask_rcnn_deepfashion2_0040.h5', '--image=.\\public\\uploads\\'+rows[0].image])
            result.stdout.on('data', function(data){
                console.log(data.toString())
                shellOutput += data;
            })
            result.stderr.on('data', function(data){
                console.log(data.toString())
            });
            result.stdout.on('end', function(){
                var output_mask = shellOutput.split("\n")[5]
                console.log(output_mask);
                console.log("save original file name: "+ rows[0].image)
                const tmp = spawn('python', ['.\\StarGAN\\save_file.py', '--image='+rows[0].image])
                tmp.stdout.on('data', function(data){
                    console.log(data.toString())
                })
                tmp.stderr.on('data', function(data){
                    console.log("original file save error stream: "+data.toString())
                })
                tmp.stdout.on('end', function(){
                        // -------------
                    //const spawn2 = require('child_process').spawn
                    const result2 = spawn('python', ['.\\StarGAN\\main.py', '--mode=test', '--dataset=RaFD', '--image_size=128', '--c_dim=5', '--rafd_image_dir=.\\StarGAN\\data\\custom\\test', '--sample_dir=.\\StarGAN\\stargan_custom\\samples', '--log_dir=.\\StarGAN\\stargan_custom\\logs', '--model_save_dir=.\\StarGAN\\stargan_custom\\models', '--result_dir=.\\public\\gan_output', '--test_iters=140000'])
                    result2.stdout.on('data', function(data){
                        console.log("result2 data stream: " + data.toString())
                    })
                    result2.stderr.on('data', function(data){
                        console.log("result2 error stream: " + data.toString())
                    })
                    result2.stdout.on('end', function(){
                        const result3 = spawn('python', ['.\\Mask_RCNN\\histogram_matching.py'])
                        result3.stdout.on('data', function(data){
                            console.log("result3 data stream: " + data.toString())
                        })
                        result3.stderr.on('data', function(data){
                            console.log("result3 error stream: " + data.toString())
                        })
                        result3.stdout.on('end', function(){
                            const result4 = spawn('python', ['.\\Mask_RCNN\\segmentation.py'])
                            result4.stdout.on('data', function(data){
                                console.log('result4 data stream: ' + data.toString())
                            })
                            result4.stderr.on('data', function(data){
                                console.log('result4 error stream: '+ data.toString())
                            })
                            result4.stdout.on('end', function(){
                                console.log('process end')
                            })
                        })
                    })

                    //------------------

                    var sql2 = "insert into outimage(id, username, image) values(?, ?, ?)";
                    conn.query(sql2, [rows[0].id, req.user.username, output_mask], (err, rows2)=>{
                        if(err) console.error(err);
                        else{
                            console.log("P output_mask: "+ output_mask);
                            console.log("P rows[0].id: "+rows[0].id)
                            console.log("P req.user.username: "+ req.user.username)
                            var sql3 = "select real_id, id, image from outimage where image = ? and id=? and username=?";
                            conn.query(sql3, [output_mask, rows[0].id, req.user.username], (err, rows3)=>{
                                console.log("rows3[0].real_id: " + rows3[0].real_id)
                                console.log("rows3[0].id: " + rows3[0].id)
                                console.log("rows3[0].image: " + rows3[0].image)
                                res.render('process', {orgimg: rows[0], newimg: rows3[0]}) // info.id, info.image로 access
                            })

                        }
                    })
                })
                
            })
        }
    })
})

const download = require('images-downloader').images;
app.get('/album/:id', (req, res)=>{
    var id = req.params.id
    const dest = "./tmp";
    var sql = "select image from outimage where real_id = ?";
    conn.query(sql, [id], (err, rows)=>{
        console.log("rows:", rows);
        var image_path = "http://localhost:3003/output/"+rows.image;
        const images = [image_path];
        download(images, dest).then(result=>{
            console.log('Images downloaded', result);
        }).catch(error => console.log('downloaded error', error))
        res.redirect('/upload')
    })
})