import * as THREE from './vendor/three.module.js';
export function startMolecule(raw){
if(window.disposeMolecularScene)window.disposeMolecularScene();
const host=document.querySelector('#molecular-stage');if(!host)return;host.dataset.renderState='initializing';
const reduced=matchMedia('(prefers-reduced-motion: reduce)');let moving=!reduced.matches,visible=true,raf=0,last=0,time=0,disposed=false;
const toggle=document.getElementById('motion-control');

let renderer;
try{renderer=new THREE.WebGLRenderer({alpha:true,antialias:true,powerPreference:'low-power'});}catch(e){host.classList.add('static-molecule');if(toggle)toggle.hidden=true;return;}
renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.7));renderer.setClearColor(0x768c8f,0);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.3;
host.dataset.renderState='webgl';host.appendChild(renderer.domElement);renderer.domElement.setAttribute('aria-hidden','true');
const scene=new THREE.Scene();scene.fog=new THREE.FogExp2(0x93aaa9,.008);
const camera=new THREE.PerspectiveCamera(38,1,.1,600);camera.position.set(0,0,95);
scene.add(new THREE.HemisphereLight(0xeef7f1,0x243e42,2.5));
const key=new THREE.DirectionalLight(0xfff3dc,5);key.position.set(-25,35,40);scene.add(key);
const rim=new THREE.DirectionalLight(0xcee5ed,4);rim.position.set(25,-5,-25);scene.add(rim);
const fill=new THREE.PointLight(0xe9ab8d,70,150,1.5);fill.position.set(10,-25,25);scene.add(fill);
const sphere=new THREE.SphereGeometry(1,18,14),cylinder=new THREE.CylinderGeometry(1,1,1,7),dummy=new THREE.Object3D();
const colors={C:0xc5d1c6,O:0xb36e51,N:0x698ba0,P:0xd7b67e};
const atomMaterial=new THREE.MeshStandardMaterial({roughness:.29,metalness:.16});
const bondMaterial=new THREE.MeshStandardMaterial({color:0xa5b5ae,roughness:.48,metalness:.10});
function molecule(){
 const model=new THREE.Group(),atoms=new THREE.InstancedMesh(sphere,atomMaterial,raw.atoms.length),bonds=new THREE.InstancedMesh(cylinder,bondMaterial,raw.bonds.length);
 raw.atoms.forEach((atom,i)=>{dummy.position.fromArray(atom.p);dummy.rotation.set(0,0,0);const radius=atom.e==='P'?.9:atom.e==='C'?.77:.68;dummy.scale.setScalar(radius);dummy.updateMatrix();atoms.setMatrixAt(i,dummy.matrix);atoms.setColorAt(i,new THREE.Color(colors[atom.e]||0xc5d1c6));});
 const up=new THREE.Vector3(0,1,0);
 raw.bonds.forEach(([a,b],i)=>{const p=new THREE.Vector3(...raw.atoms[a].p),q=new THREE.Vector3(...raw.atoms[b].p),delta=q.clone().sub(p);dummy.position.copy(p).add(q).multiplyScalar(.5);dummy.quaternion.setFromUnitVectors(up,delta.clone().normalize());dummy.scale.set(.20,delta.length(),.20);dummy.updateMatrix();bonds.setMatrixAt(i,dummy.matrix)});
 model.add(atoms,bonds);return model;
}
const main=molecule();main.rotation.set(.16,-.55,-.60);main.position.set(5,1,0);main.scale.setScalar(1.33);scene.add(main);
const back=molecule();back.position.set(-39,10,-48);back.rotation.set(.7,.6,.42);back.scale.setScalar(1.9);scene.add(back);
const distant=molecule();distant.position.set(46,20,-70);distant.rotation.set(-.4,.8,-.8);distant.scale.setScalar(2.1);scene.add(distant);
const dustGeo=new THREE.BufferGeometry(),dust=[];for(let i=0;i<95;i++)dust.push(Math.sin(i*7.3)*90,Math.cos(i*3.7)*60,-20-(i%12)*10);
dustGeo.setAttribute('position',new THREE.Float32BufferAttribute(dust,3));const dustMat=new THREE.PointsMaterial({color:0xf3f1df,size:.22,transparent:true,opacity:.5});scene.add(new THREE.Points(dustGeo,dustMat));
let px=0,py=0;
function pointer(e){if(!moving)return;const r=host.getBoundingClientRect();px=(e.clientX-r.left)/r.width-.5;py=(e.clientY-r.top)/r.height-.5;}
function resize(){const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.position.z=w<600?123:95;camera.updateProjectionMatrix();render();}
function render(){main.rotation.y=-.55+time*.10+px*.10;main.rotation.x=.16+py*.09;main.position.y=1+Math.sin(time*.22)*1.2;back.rotation.y=.6-time*.035;renderer.render(scene,camera)}
function frame(now){raf=0;if(disposed||!host.isConnected)return;if(!visible||document.hidden||!moving)return;if(now-last>32){time+=Math.min((now-last)/1000,.07);last=now;render()}raf=requestAnimationFrame(frame)}
function start(){if(!raf&&moving&&visible&&!document.hidden){last=performance.now();raf=requestAnimationFrame(frame)}}
function motion(){moving=!moving&&!reduced.matches;toggle.textContent=moving?'Pause motion Ⅱ':'Resume motion ▷';toggle.setAttribute('aria-pressed',String(!moving));start()}
function preference(){moving=!reduced.matches;toggle.textContent=moving?'Pause motion Ⅱ':'Motion reduced';render();start()}
const ro=new ResizeObserver(resize);ro.observe(host);const io=new IntersectionObserver(([e])=>{visible=e.isIntersecting;start()});io.observe(host);
host.parentElement.addEventListener('pointermove',pointer);document.addEventListener('visibilitychange',start);reduced.addEventListener('change',preference);toggle?.addEventListener('click',motion);
window.disposeMolecularScene=()=>{disposed=true;cancelAnimationFrame(raf);ro.disconnect();io.disconnect();host.parentElement?.removeEventListener('pointermove',pointer);document.removeEventListener('visibilitychange',start);reduced.removeEventListener('change',preference);toggle?.removeEventListener('click',motion);sphere.dispose();cylinder.dispose();atomMaterial.dispose();bondMaterial.dispose();dustGeo.dispose();dustMat.dispose();renderer.dispose();renderer.domElement.remove()};
resize();preference();
}
