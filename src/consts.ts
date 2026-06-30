// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.
import type { Header, Footer, About, Project } from "./types.ts";

import ProfilePic from "./assets/avatar.png";

import PenweaveImage from "./assets/projects/penweave.png";
import MGUScraperImage from "./assets/projects/mguscraper.png";
import FlashifyImage from "./assets/projects/flashify.png";
import WeatherifyImage from "./assets/projects/weatherify.png";
import BriefImage from "./assets/projects/brief.svg";
import DefuseImage from "./assets/projects/defuse.svg";
import LipiImage from "./assets/projects/lipi.svg";
import ExchangeImage from "./assets/projects/exchange.svg";
import PlaceholderImage from "./assets/projects/placeholder.svg";


// education

import ETHLogo from "./assets/education/eth.png";
import PaduaLogo from "./assets/education/padua.png";
import BexhillLogo from "./assets/education/bexhill.jpg";


export const SITE_URL: string = "https://simone-piccinini.github.io";



export const ENABLE_STARDUST_BG: boolean = false;

export const meta = {
  about: {
    // index page
    title: "Simone Piccinini",
    description:
      "Compute engineering student at the univeristy of Padua.",
  },
  projects: {
    title: "Projects",
    description:
      "",
  },
  // blog post title and description are taken from the variables in markdown file
};

export const header: Header = { logoTitle: "SP" };

export const footer: Footer = {
  // parses html
  content:
    "",
};

export const about: About = {
  // parses html
  headLine:
    "Hi, I'm <span class='fancy-highlight font-black'>Simone Piccinini</span>",
  tagLine: "Computer engineer student",
  profilePic: ProfilePic,
  // parses html
  academicBio: "I'm 21 years old and currently studying Computer Engineering at the University of Padua. <br> My work and interests focus on the intersection of <b class='text-base-content'>mathematics</b>, <b class='text-base-content'>physics</b>, and <b class='text-base-content'>machine learning</b>, love creating things.",
  personalBio: "Beyond my studies, I am a dedicated musician 🎸🎹 and a sportsperson 🎾 🏃‍♂️.",
  links: [
    // Lucide icons
    { icon: "Github", href: "https://github.com/dreaazy" },
    { icon: "Linkedin", href: "https://www.linkedin.com/in/simone-piccinini-b32966261/" },
  ],
  resumeHref:
    "https://drive.google.com/file/d/10dfGCIiX2b7Wf-Lj51ypt3UHgPiPQ3ZV/view?usp=sharing",
  workExperience: [
    {
      title: "Full Stack Developer Intern",
      timeline: "May 2024 - Oct 2024",
      company: "Rabbitsquare • India",
      description:
        "Developed a LAMP stack web solution for a Civil Service academy, including a customizable public website, management portal, user portal, and exam result publishing system.",
    },
  ],
  education: [
    {
      title: "Incoming Erasmus student",
      timeline: "Sep 2026 - Feb 2027",
      institution: "ETH Zurich",
      description:
        "Probabilistic AI, Mathematical optimization",
      logo: ETHLogo,

  
    },
    {
      title: "Bachelor of Computer Engineering",
      timeline: "Jun 2024 - Current",
      institution: "University of Padua",
      description: "",
      logo: PaduaLogo,

    }
    ,
    {
      title: "Exchange student period",
      timeline: "Jan 2023 - June 2023",
      institution: "Bexhill college",
      description: "Mathematics, Economoics, Computer science",
      logo: BexhillLogo,

    }
    
  ],
  // parses html
  getInTouch:
    "Drop me an email at <a href='mailto:piccinini.simone2005@gmail.com' class='primary-underline'>piccinini.simone2005@gmail.com</a>",
};



// add blog articles in /src/content/blog
