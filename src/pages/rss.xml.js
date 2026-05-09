import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { meta } from "../consts";

export async function GET(context) {
  const projectEntries = await getCollection("projects");
  const articleEntries = await getCollection("articles");

  // Combine both collections into one feed
  const allPosts = [...projectEntries, ...articleEntries].sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf()
  );

  return rss({
    title: meta.about.title,
    description: meta.about.description,
    site: context.site,
    items: allPosts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.pubDate,
      description: post.data.description,
      // Logic for URLs: projects use /projects/ path, articles use /articles/
      link: post.collection === 'projects' 
        ? `/projects/${post.id.replace('/index', '')}/` 
        : `/articles/${post.id}/`,
    })),
  });
}